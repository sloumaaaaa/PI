# main.py (Streamlit App)
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True' # Add this line AT THE VERY TOP
import streamlit as st
from dotenv import load_dotenv
import numpy as np  # Add this import at the top if not already present


# Load environment variables (e.g., OPENAI_API_KEY)
load_dotenv()

# --- Import RAG Modules ---
from acq_agent.build_faiss import (
    extract_text_from_pdf_bytes, extract_text_from_docx_bytes,
    extract_text_from_html_bytes,
    process_and_index_document_in_memory,
    save_faiss_index_and_metadata, load_faiss_index_and_metadata_from_files,
    EMBEDDING_DIMS, DEFAULT_EMBEDDING_MODEL
)

from exp_agent.generator import build_exam_generation_prompt_with_syllabus, generate_exam_from_course
from exp_agent.embedding import vectorize_query_text
from exp_agent.chunk_retriever import retrieve_top_k_chunks_from_memory

# --- Streamlit Page Configuration ---
st.set_page_config(page_title="ESB Exam Gen", page_icon="🧩", layout="wide")
st.markdown("<h1 style='text-align: center;'>ESB Exam Gen</h1>", unsafe_allow_html=True)
st.markdown("---")

# --- Initialize Session State (more comprehensive) ---
def init_session_state():
    defaults = {
        "openai_api_key": os.getenv("OPENAI_API_KEY"),
        "embedding_model": DEFAULT_EMBEDDING_MODEL,
        "rag_enabled": True,
        "faiss_index": None, # Will hold the FAISS index object
        "faiss_index_dim": None, # Stores dimension of the current index
        "chunk_store": [],   # List of dicts, metadata for each chunk
        "doc_counter": 0,    # For generating unique doc IDs
        "chat_history": [],  # For storing conversation
        # Keys for resetting file uploaders/inputs
        "file_uploader_key": 0,
        "url_input_key": 0,
        "text_area_key": 0,
        # For loading/saving index (optional)
        "index_loaded_from_file": False
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# --- Sidebar Configuration ---
with st.sidebar:
    st.header("⚙️ Configuration")

    # API Key
    api_key_input = st.text_input(
        "Clé API OpenAI:",
        type="password",
        value=st.session_state.openai_api_key or "",
        help="Votre clé API OpenAI. Peut aussi être définie via la variable d'environnement OPENAI_API_KEY."
    )
    if api_key_input:
        st.session_state.openai_api_key = api_key_input

    # Embedding Model
    prev_embedding_model = st.session_state.embedding_model
    st.session_state.embedding_model = st.selectbox(
        "Modèle d'Embedding OpenAI:",
        list(EMBEDDING_DIMS.keys()),
        index=list(EMBEDDING_DIMS.keys()).index(st.session_state.embedding_model),
        help="Changer de modèle effacera l'index existant si les dimensions diffèrent."
    )
    if st.session_state.embedding_model != prev_embedding_model:
        st.warning("Modèle d'embedding changé. L'index FAISS sera réinitialisé au prochain traitement de document.")
        st.session_state.faiss_index = None
        st.session_state.chunk_store = []
        st.session_state.faiss_index_dim = None
        st.session_state.index_loaded_from_file = False # Reset flag

    st.info(f"Modèle actif : `{st.session_state.embedding_model}`")

    # RAG Toggle
    st.session_state.rag_enabled = st.checkbox("Activer RAG (Recherche Augmentée)", value=st.session_state.rag_enabled)

   
    # --- Load/Save Index (Optional) ---
    st.subheader("Gestion de l'Index FAISS")
    if st.button("Sauvegarder l'Index Actif", help="Sauvegarde l'index en mémoire vers data/index.faiss et data/chunks_metadata.json"):
        if st.session_state.faiss_index and st.session_state.chunk_store:
            save_faiss_index_and_metadata(st.session_state.faiss_index, st.session_state.chunk_store)
            st.success("Index et métadonnées sauvegardés dans `data/`.")
        else:
            st.warning("Aucun index en mémoire à sauvegarder.")

    if st.button("Charger l'Index depuis Fichiers", help="Charge data/index.faiss et data/chunks_metadata.json. Écrase l'index en mémoire."):
        loaded_index, loaded_metadata = load_faiss_index_and_metadata_from_files()
        if loaded_index and loaded_metadata:
            st.session_state.faiss_index = loaded_index
            st.session_state.chunk_store = loaded_metadata
            st.session_state.faiss_index_dim = loaded_index.d
            # Try to infer embedding model from dimension, or warn user
            matched_model = False
            for model, dim in EMBEDDING_DIMS.items():
                if dim == loaded_index.d:
                    st.session_state.embedding_model = model
                    matched_model = True
                    break
            if matched_model:
                st.success(f"Index chargé. Modèle d'embedding ajusté à '{st.session_state.embedding_model}'.")
            else:
                st.warning(f"Index chargé (dim: {loaded_index.d}), mais aucun modèle d'embedding correspondant trouvé. Veuillez sélectionner un modèle compatible.")
            st.session_state.index_loaded_from_file = True
            st.rerun() # Refresh UI with new index info
        else:
            st.error("Échec du chargement de l'index depuis les fichiers.")
    st.caption(f"Index en mémoire: {'Chargé depuis fichier' if st.session_state.index_loaded_from_file else 'Construit dynamiquement'}")


# --- Main Application Area ---

# 1. Agent d'Acquisition
st.header("1. Acquisition de Documents")
# Onglet unique : Fichier uniquement
st.markdown("**Seuls les fichiers (PDF, DOCX, HTML, TXT) sont acceptés pour l'indexation.**")
doc_acquisition_tabs = st.tabs(["📁 Fichier"])

doc_text_to_process = None
doc_name = "Document Inconnu"
source_type = "inconnu"
input_source_used = None # For resetting inputs

with doc_acquisition_tabs[0]: # File Upload
    uploaded_file = st.file_uploader(
        "Choisissez un fichier (PDF, DOCX, HTML, TXT):",
        type=["pdf", "docx", "html", "txt"],
        key=f"file_uploader_{st.session_state.file_uploader_key}"
    )
    if uploaded_file:
        file_bytes = uploaded_file.getvalue()
        doc_name = uploaded_file.name
        source_type = f"fichier: {uploaded_file.type}"
        input_source_used = "file"
        if uploaded_file.type == "application/pdf":
            doc_text_to_process = extract_text_from_pdf_bytes(file_bytes)
        elif uploaded_file.type in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document", "application/msword"]:
            doc_text_to_process = extract_text_from_docx_bytes(file_bytes)
        elif uploaded_file.type == "text/html":
            doc_text_to_process = extract_text_from_html_bytes(file_bytes.decode('utf-8', errors='ignore')) # Assuming html is text
        elif uploaded_file.type == "text/plain":
            doc_text_to_process = file_bytes.decode('utf-8', errors='ignore')
        else:
            st.warning(f"Type de fichier non supporté pour l'extraction directe: {uploaded_file.type}")

# Chunking Method
col_chunk1, col_chunk2 = st.columns(2)
with col_chunk1:
    chunk_method_selected = st.selectbox("Méthode de Découpage:", ["Paragraphe ('\\n\\n')", "Taille Fixe (Mots)"])
with col_chunk2:
    chunk_size_words_val = 150
    chunk_overlap_words_val = 20
    if chunk_method_selected == "Taille Fixe (Mots)":
        chunk_size_words_val = st.slider("Taille Chunks (mots):", 50, 500, 150)
        chunk_overlap_words_val = st.slider("Chevauchement (mots):", 0, 100, 20)

if st.button("✨ Traiter et Indexer le Document"):
    if not st.session_state.openai_api_key:
        st.error("Veuillez configurer votre clé API OpenAI dans la barre latérale.")
    elif doc_text_to_process and doc_text_to_process.strip():
        st.session_state.doc_counter += 1
        with st.spinner(f"Traitement de '{doc_name}' et indexation FAISS..."):
            new_faiss_index, new_chunk_store, chunks_added = process_and_index_document_in_memory(
                doc_text_to_process=doc_text_to_process,
                doc_name_for_processing=doc_name,
                source_type_for_processing=source_type,
                openai_api_key=st.session_state.openai_api_key,
                embedding_model_name=st.session_state.embedding_model,
                chunk_method=chunk_method_selected,
                chunk_size_words=chunk_size_words_val,
                chunk_overlap_words=chunk_overlap_words_val,
                current_faiss_index=st.session_state.faiss_index,
                current_chunk_store=st.session_state.chunk_store,
                doc_counter=st.session_state.doc_counter
            )
            if chunks_added > 0:
                st.session_state.faiss_index = new_faiss_index
                st.session_state.chunk_store = new_chunk_store # process_and_index appends, so this is fine
                st.session_state.faiss_index_dim = new_faiss_index.d
                # --- Save index and metadata to disk ---
                save_faiss_index_and_metadata(new_faiss_index, new_chunk_store)
                st.success(f"{chunks_added} chunks de '{doc_name}' ajoutés à l'index. Total : {st.session_state.faiss_index.ntotal} chunks.")
                # Reset the input that was used
                if input_source_used == "file" or input_source_used == "image_file": st.session_state.file_uploader_key += 1
                elif input_source_used == "url": st.session_state.url_input_key += 1
                elif input_source_used == "text_area": st.session_state.text_area_key += 1
                st.session_state.index_loaded_from_file = False # New content added dynamically
                st.rerun()
            else:
                st.warning(f"Aucun chunk n'a été ajouté pour '{doc_name}'. Vérifiez le contenu ou les logs.")
    else:
        st.warning("Veuillez fournir un document (fichier, texte, URL ou image) à traiter.")

# Display Indexed Chunks Overview
st.subheader(" Aperçu de l'Index")
if st.session_state.faiss_index and st.session_state.faiss_index.ntotal > 0:
    st.info(f"**{st.session_state.faiss_index.ntotal}** chunks indexés (Dimension: {st.session_state.faiss_index_dim}). "
            f"Métadonnées pour **{len(st.session_state.chunk_store)}** chunks.")
    if st.session_state.faiss_index.ntotal != len(st.session_state.chunk_store):
        st.warning("Incohérence: Nombre de vecteurs FAISS != nombre de métadonnées de chunks.")
    
    with st.expander("Voir les derniers chunks indexés (aperçu)"):
        num_to_show = min(5, len(st.session_state.chunk_store))
        if num_to_show > 0:
            preview_data = [{
                "Nom Doc": c.get('doc_name', 'N/A'),
                "Source": c.get('metadata',{}).get('source_type', 'N/A'),
                "Langue (Doc)": c.get('metadata',{}).get('lang','N/A'),
                "Aperçu": c['text'][:100] + "..." if len(c['text']) > 100 else c['text']
            } for c in reversed(st.session_state.chunk_store[-num_to_show:])]
            st.dataframe(preview_data, use_container_width=True)
else:
    st.info("Aucun document n'a encore été indexé. Ajoutez des documents pour commencer.")

st.markdown("---")

# --- Syllabus Upload Section ---
st.header("2. Syllabus (obligatoire pour la génération d'examen)")
syllabus_file = st.file_uploader(
    "Uploader le fichier Syllabus (PDF, DOCX, TXT, etc.) :",
    type=["pdf", "docx", "txt"],
    key="syllabus_file_uploader"
)
syllabus_text = ""
if syllabus_file:
    file_bytes = syllabus_file.getvalue()
    if syllabus_file.type == "application/pdf":
        syllabus_text = extract_text_from_pdf_bytes(file_bytes)
    elif syllabus_file.type in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document", "application/msword"]:
        syllabus_text = extract_text_from_docx_bytes(file_bytes)
    elif syllabus_file.type == "text/plain":
        syllabus_text = file_bytes.decode('utf-8', errors='ignore')
    else:
        st.warning(f"Type de fichier non supporté pour le syllabus: {syllabus_file.type}")
    st.session_state.syllabus_text = syllabus_text
elif "syllabus_text" in st.session_state:
    syllabus_text = st.session_state.syllabus_text
else:
    st.info("Aucun syllabus chargé.")

# --- Exam Generator UI ---
st.header("Générateur d'Examens à partir d'un Document Indexé et d'un Syllabus")

# Sélection du document (chunk) à utiliser pour générer l'examen
if st.session_state.chunk_store:
    doc_options = list({c['doc_name'] for c in st.session_state.chunk_store})
    selected_doc = st.selectbox("Sélectionnez le document source pour l'examen :", doc_options)
    # Concaténer tous les chunks du document sélectionné
    selected_chunks = [c['text'] for c in st.session_state.chunk_store if c['doc_name'] == selected_doc]
    course_text = "\n".join(selected_chunks)
else:
    st.warning("Aucun document indexé disponible pour générer un examen.")
    course_text = ""

col1, col2, col3, col4, col5, col6 = st.columns(6)
with col1:
    num_qcm = st.number_input("Nombre de QCM", min_value=0, max_value=50, value=2)
with col2:
    num_ouvertes = st.number_input("Nombre d'ouvertes", min_value=0, max_value=50, value=2)
with col3:
    num_vf = st.number_input("Nombre de vrai/faux", min_value=0, max_value=50, value=1)
with col4:
    num_assoc = st.number_input("Nombre d'associations", min_value=0, max_value=50, value=0)
with col5:
    num_trous = st.number_input("Nombre de textes à trous", min_value=0, max_value=50, value=0)
with col6:
    difficulty = st.selectbox("Difficulté", ["facile", "moyen", "difficile"])

total_questions = num_qcm + num_ouvertes + num_vf + num_assoc + num_trous

# Construction du dictionnaire question_types_dict AVANT toute utilisation
question_types_dict = {}
if num_qcm > 0:
    question_types_dict['QCM'] = num_qcm
if num_ouvertes > 0:
    question_types_dict['ouvertes'] = num_ouvertes
if num_vf > 0:
    question_types_dict['vrai/faux'] = num_vf
if num_assoc > 0:
    question_types_dict['association'] = num_assoc
if num_trous > 0:
    question_types_dict['texte à trous'] = num_trous

generated_exam = None
final_response_for_chat = None  # <-- Ajout explicite pour éviter affichage prématuré

# Get user input (make sure this is before the function call)
user_query = st.text_area("Votre question :", key="user_query_input")

if st.button("Générer l'examen", type="primary"):
    if not syllabus_text.strip():
        st.warning("Veuillez fournir un syllabus pour générer l'examen.")
    elif not course_text.strip():
        st.warning("Aucun texte de cours disponible pour générer l'examen.")
    elif total_questions == 0:
        st.warning("Veuillez demander au moins une question.")
    else:
        # Limiter la taille du syllabus et du cours AVANT génération du prompt pour éviter l'erreur Groq
        max_syllabus_chars = 2000
        max_course_chars = 2000
        syllabus_text_trunc = syllabus_text[:max_syllabus_chars] + '\n[Texte tronqué]' if len(syllabus_text) > max_syllabus_chars else syllabus_text
        course_text_trunc = course_text[:max_course_chars] + '\n[Texte tronqué]' if len(course_text) > max_course_chars else course_text
        with st.spinner("Génération de l'examen en cours..."):
            prompt_context = build_exam_generation_prompt_with_syllabus(
                syllabus_text=syllabus_text_trunc,
                course_text=course_text_trunc,
                question_types=question_types_dict,
                difficulty=difficulty
            )
            generated_exam = generate_exam_from_course(
                course_text=prompt_context,
                num_questions=total_questions,
                question_type=question_types_dict,
                difficulty=difficulty
            )
            final_response_for_chat = generated_exam  # <-- Stocke la réponse seulement si bouton cliqué

# --- Pipeline complet RAG + CoNLI (exemple détaillé) ---
retrieved_chunks = []
pipeline_details_md = ""
# Empêche toute génération pipeline tant que l'utilisateur n'a pas cliqué sur le bouton
if final_response_for_chat:
    if st.session_state.rag_enabled:
        with st.spinner("Recherche et génération de la réponse..."):
            # 2. Vectorisation de la requête
            query_embedding = vectorize_query_text(
                cleaned_query_text=syllabus_text,  # ou autre texte de requête pertinent
                embedding_model=st.session_state.embedding_model
            )
            if query_embedding is not None and hasattr(query_embedding, "__len__") and len(query_embedding) > 0:
                pipeline_details_md += "**2. Vectorisation Requête:** Succès\n\n"
                top_k_retrieval = 3
                similarity_threshold_val = 0.3
                retrieved_chunks = retrieve_top_k_chunks_from_memory(
                    query_embedding=query_embedding,
                    faiss_index_in_memory=st.session_state.faiss_index,
                    chunk_store_in_memory=st.session_state.chunk_store,
                    top_k=top_k_retrieval,
                    similarity_threshold=similarity_threshold_val
                )
                pipeline_details_md += f"**3. Récupération Chunks:** {len(retrieved_chunks)} chunk(s) pertinent(s) trouvé(s).\n"
                if retrieved_chunks:
                    for i_rc, rc in enumerate(retrieved_chunks):
                        pipeline_details_md += (f"   - Chunk {i_rc+1}: Doc='{rc.get('doc_name', 'N/A')}', "
                                                f"Score={rc.get('similarity_score', 0.0):.4f}, "
                                                f"Texte='{rc['text'][:60]}...'\n")
                pipeline_details_md += "\n"
            else:
                pipeline_details_md += "**2. Vectorisation Requête:** Échec\n\n"
                pipeline_details_md += "**Mode RAG actif, mais aucun document indexé.** La réponse sera basée sur les connaissances générales du LLM.\n\n"

            # 4. Construction du Prompt Augmenté
            augmented_prompt = build_exam_generation_prompt_with_syllabus(
                syllabus_text=syllabus_text,
                course_text=course_text,
                question_types=question_types_dict,
                difficulty=difficulty
            )
            pipeline_details_md += f"**4. Prompt Augmenté (aperçu):**\n```markdown\n{augmented_prompt[:500]}...\n```\n\n"

            # 5. Génération de la Réponse LLM
            raw_llm_response = generate_exam_from_course(
                course_text=augmented_prompt,
                num_questions=total_questions,
                question_type=question_types_dict,
                difficulty=difficulty
            )
            pipeline_details_md += f"**5. Réponse Brute LLM (aperçu):**\n`{raw_llm_response[:100]}...`\n\n"

            # 6. Post-traitement de la Réponse
            query_keywords_for_highlight = syllabus_text.split()[:5]
            from exp_agent.response_postprocessor import postprocess_llm_response
            final_response_for_chat, postprocessing_flags = postprocess_llm_response(
                raw_llm_response=raw_llm_response,
                user_query=user_query,
                retrieved_chunks=retrieved_chunks,
                faiss_index_path="data/index.faiss",
                metadata_path="data/chunks_metadata.json",
                guardrails_enabled={"conli": False}  # <--- Add this line
            )

            # Affichage détaillé de toutes les étapes du post-processing
            pipeline_details_md += "### Post-traitement & Analyse Qualité\n\n"
            pipeline_details_md += "**Analyse qualité LLM (Heuristique via prompt)**\n"
            for key in ["is_toxic", "is_uncertain", "has_hallucination", "is_factually_incorrect", "is_answer_acceptable"]:
                value = postprocessing_flags.get(key)
                if value is not None:
                    icon = "✔" if value is False else "⚠️" if value is True else "ℹ️"
                    pipeline_details_md += f"- `{key}`: {icon} `{value}`\n"
            pipeline_details_md += "\n"
            # Analyse CoNLI
            if "conli_detected_hallucination" in postprocessing_flags:
                if postprocessing_flags["conli_detected_hallucination"]:
                    pipeline_details_md += "** CoNLI Validator:** Hallucination détectée ⚠️\n"
                    pipeline_details_md += f"- Correction proposée :\n```markdown\n{postprocessing_flags.get('conli_fix_value', '')}\n```\n"
                else:
                    pipeline_details_md += "** CoNLI Validator:** ✅ Aucun problème détecté.\n"
            pipeline_details_md += "\n"
            # Aperçu de la réponse finale avec mise en forme
            pipeline_details_md += "**Réponse Finale Formatée (aperçu):**\n"
            pipeline_details_md += f"```markdown\n{final_response_for_chat[:500]}...\n```\n"
    else:
        pipeline_details_md += "**Mode RAG désactivé.** La réponse sera basée sur les connaissances générales du LLM.\n\n"

# Affichage assistant
if final_response_for_chat:
    st.subheader("Réponse de l'assistant :")
    st.markdown(final_response_for_chat)
    with st.expander("Voir les détails du pipeline RAG pour cette réponse"):
        st.markdown(pipeline_details_md)

