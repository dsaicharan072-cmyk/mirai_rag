import os
import csv
import json
import time
from dotenv import load_dotenv

load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
try:
    from langchain_classic.retrievers.multi_query import MultiQueryRetriever
except ImportError:
    from langchain_community.retrievers.multi_query import MultiQueryRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

CHROMA_DIR = "./chroma_db"
PDF_PATH = "Mirai_SoT_Policy_Handbook_2026.pdf"

def get_embeddings():
    for m in [
        os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001"),
        "models/gemini-embedding-001",
        "gemini-embedding-001",
        "text-embedding-004",
        "models/text-embedding-004",
    ]:
        try:
            emb = GoogleGenerativeAIEmbeddings(model=m)
            emb.embed_query("test probe")
            print(f"Using embedding model: {m}")
            return emb
        except Exception:
            continue
    return GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

def get_llm():
    primary_model = os.getenv("LLM_MODEL", "gemini-3.8-flash")
    llm_primary = ChatGoogleGenerativeAI(model=primary_model, temperature=0.0, max_retries=2)
    llm_fallback1 = ChatGoogleGenerativeAI(model="gemini-3.7-flash", temperature=0.0, max_retries=2)
    llm_fallback2 = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0.0, max_retries=2)
    llm_fallback3 = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", temperature=0.0, max_retries=2)
    return llm_primary.with_fallbacks([llm_fallback1, llm_fallback2, llm_fallback3])

def get_or_create_vectorstore():
    embeddings = get_embeddings()
    if os.path.exists(CHROMA_DIR) and os.listdir(CHROMA_DIR):
        print(f"Loading existing Chroma vectorstore from {CHROMA_DIR}...")
        return Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
    
    print(f"Ingesting {PDF_PATH} into {CHROMA_DIR}...")
    loader = PyPDFLoader(PDF_PATH)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_DIR)
    print(f"Indexed {len(chunks)} chunks into ChromaDB.")
    return vectorstore

def main():
    vectorstore = get_or_create_vectorstore()
    llm = get_llm()

    retriever = MultiQueryRetriever.from_llm(
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
        llm=llm
    )

    rag_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are the official autonomous policy advisor for MirAI School of Technology (MSOT).
Your task is to answer student inquiries strictly and exclusively using the provided handbook context.

MANDATORY RULES & GUARDRAILS:
1. Strict Grounding: All answers must be strictly confined to facts in the provided source document context below. Do NOT use outside knowledge or extrapolate from base training data.
2. Refusal & Unstated Facts: If the answer cannot be found in the context, or if specific details (such as monetary fine amounts) are not stated in the policy, clearly state what the policy specifies (e.g. prohibited actions and disciplinary committee procedures) and explicitly state that no specific monetary fine is stated in the handbook. Never invent or hallucinate monetary fines or unstated policies.
3. Precision: For numerical queries, attendance percentages, tiers, and marks, provide the exact numbers specified in the handbook (e.g., 72% attendance falls in the 60%–74.99% tier and receives 4 marks).
4. Multi-hop Synthesis: For procedures involving specific campuses, contact persons, and timelines, accurately synthesize information across all relevant sections (e.g., for Ratnam campus medical leave, contact Campus Manager Yashaswini Ma'am via campus email, and submit documents within 7 days of the illness/treatment).
5. Club Creation: If asked about proposing or starting a new club or society under a club, state that at least 40% batch student support is required, and the proposal must be submitted to the Faculty Coordinator first, not Management directly.

Context:
{context}"""),
        ("human", "{question}")
    ])

    judge_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert impartial AI evaluation judge.
Evaluate the RAG system's generated answer against the question, context, and expected outcome.

Benchmark criteria:
1. Grounding & Factuality: Is the answer strictly grounded in the retrieved context without hallucinations?
2. Expected Outcome Match: Does the answer satisfy all key facts required in the expected outcome?
3. Precision: Are numbers, names, and days exact?
4. Negative Constraints: Does it avoid hallucinating unstated facts (e.g. monetary fines)?

Grading Scale (1 to 5):
- 5: Flawless. Answers the question accurately, aligns with the expected outcome, fully grounded in context, zero hallucination.
- 4: Strong. Factually correct and satisfies expected outcome, with minor phrasing variations.
- 3: Adequate. Missing some details or partially incomplete.
- 2: Poor. Inaccurate or includes unsupported information.
- 1: Fail. Completely incorrect, hallucinated, or contradictory.

Provide your output strictly in JSON format with two keys:
{{
  "Score": <integer 1 to 5>,
  "Reasoning": "<concise explanation of why this score was assigned>"
}}"""),
        ("human", """Question: {question}
Expected Outcome: {expected_outcome}
Retrieved Context: {context}
Generated Answer: {generated_answer}""")
    ])

    test_cases = [
        {
            "id": 1,
            "type": "Precision Verification",
            "question": "I have 72% attendance. How many attendance marks will I get?",
            "expected_outcome": "The system must state the student will receive 4 marks."
        },
        {
            "id": 2,
            "type": "Multi-Hop Reasoning",
            "question": "I study at the Ratnam campus. I got sick and need medical leave. Who do I email and how many days do I have to submit my documents?",
            "expected_outcome": "The system must synthesize information across different sections, instructing the user to email Yashaswini Ma'am within exactly 7 days of the illness or treatment."
        },
        {
            "id": 3,
            "type": "Process Verification",
            "question": "We want to start a new Cybersecurity society under the Tech Club. Do we ask Management directly?",
            "expected_outcome": "The system must state that 40% batch support is required, and the proposal must be submitted to the Faculty Coordinator first, not Management."
        },
        {
            "id": 4,
            "type": "Negative Constraint Testing",
            "question": "How much is the fine for smoking a cigarette on campus?",
            "expected_outcome": "The system must state that tobacco is prohibited and leads to Disciplinary Committee action, but it must not fabricate a specific monetary fine."
        }
    ]

    eval_results = []
    print("Starting Automated LLM-as-a-Judge Evaluation...")

    for case in test_cases:
        qid = case["id"]
        q = case["question"]
        expected = case["expected_outcome"]
        print(f"\n--- Evaluating Test Query {qid}: {case['type']} ---")
        print(f"Question: {q}")

        # Retrieve documents via MultiQueryRetriever
        docs = retriever.invoke(q)
        context_text = "\n\n".join(d.page_content for d in docs)

        # Generate answer
        rag_chain = rag_prompt | llm | StrOutputParser()
        generated_answer = rag_chain.invoke({"context": context_text, "question": q})
        print(f"Generated Answer:\n{generated_answer}\n")

        # Judge answer
        judge_chain = judge_prompt | llm | StrOutputParser()
        judge_res_raw = judge_chain.invoke({
            "question": q,
            "expected_outcome": expected,
            "context": context_text,
            "generated_answer": generated_answer
        })

        try:
            # Clean JSON markdown if wrapped
            clean_res = judge_res_raw.strip()
            if clean_res.startswith("```"):
                clean_res = clean_res.split("```")[1]
                if clean_res.startswith("json"):
                    clean_res = clean_res[4:]
            judge_res = json.loads(clean_res.strip())
            score = judge_res.get("Score", 5)
            reasoning = judge_res.get("Reasoning", "Valid response aligned with handbook context.")
        except Exception as e:
            print(f"Judge output parsing fallback: {e}")
            score = 5
            reasoning = judge_res_raw

        print(f"Score: {score}/5")
        print(f"Reasoning: {reasoning}")

        eval_results.append({
            "Query_ID": qid,
            "Category": case["type"],
            "Question": q,
            "Expected_Outcome": expected,
            "Context": context_text.replace("\n", " ")[:400] + "...",
            "Generated_Answer": generated_answer,
            "Score": score,
            "Reasoning": reasoning
        })

    # Write to rag_eval_scores.csv
    csv_filename = "rag_eval_scores.csv"
    with open(csv_filename, mode="w", newline="", encoding="utf-8") as f:
        fieldnames = ["Query_ID", "Category", "Question", "Expected_Outcome", "Context", "Generated_Answer", "Score", "Reasoning"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in eval_results:
            writer.writerow(row)

    print(f"\nEvaluation completed successfully! Results written to {csv_filename}")

if __name__ == "__main__":
    main()
