from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
from mecab import MeCab
from typing import List

app = FastAPI()

# Load models
sbert_model = SentenceTransformer('distiluse-base-multilingual-cased-v2')
mecab = MeCab()

# 최소화된 동의어 사전 (핵심 쌍만 정의)
SYNONYMS = {
    "ai": "인공지능",
    "artificial intelligence": "인공지능"
}


class TextInput(BaseModel):
    reference_text: str  # 기준 텍스트 (정답)
    response_text: str  # 응답 텍스트


def extract_keywords(text: str, top_n: int = 5) -> List[str]:
    """Mecab으로 명사 및 외국어 추출"""
    parsed = mecab.pos(text)
    keywords = [word for word, pos in parsed if pos.startswith('N') or pos == 'SL']  # 명사 + 외국어(AI 등)
    # 중복 제거 및 상위 top_n 선택
    unique_keywords = list(dict.fromkeys(keywords))
    return unique_keywords[:top_n]


def normalize_keyword(keyword: str) -> str:
    """동의어 사전 적용"""
    return SYNONYMS.get(keyword.lower(), keyword)


def calculate_similarity(reference: str, response: str) -> float:
    """SBERT로 문장 유사도 계산"""
    embeddings1 = sbert_model.encode(reference, convert_to_tensor=True)
    embeddings2 = sbert_model.encode(response, convert_to_tensor=True)
    similarity = util.cos_sim(embeddings1, embeddings2)
    return similarity.item()


def evaluate_keyword_match(ref_keywords: List[str], resp_keywords: List[str], threshold: float = 0.85) -> float:
    """키워드 매칭: 동의어 사전 + SBERT 유사도"""
    if not ref_keywords:
        return 0.0

    # 정규화된 키워드
    ref_keywords_norm = [normalize_keyword(kw) for kw in ref_keywords]
    resp_keywords_norm = [normalize_keyword(kw) for kw in resp_keywords]

    matches = 0
    for ref_kw in ref_keywords_norm:
        for resp_kw in resp_keywords_norm:
            # 완전 일치 확인
            if ref_kw == resp_kw:
                matches += 1
                break
            # SBERT 유사도 계산
            emb1 = sbert_model.encode(ref_kw, convert_to_tensor=True)
            emb2 = sbert_model.encode(resp_kw, convert_to_tensor=True)
            sim = util.cos_sim(emb1, emb2).item()
            if sim >= threshold:  # 유사도 임계값
                matches += 1
                break

    total_possible = len(ref_keywords)
    return matches / total_possible if total_possible > 0 else 0.0


@app.post("/evaluate")
async def evaluate_text(input_data: TextInput):
    try:
        # 키워드 추출
        ref_keywords = extract_keywords(input_data.reference_text)
        resp_keywords = extract_keywords(input_data.response_text)

        # 점수 계산
        similarity_score = calculate_similarity(
            input_data.reference_text,
            input_data.response_text
        )

        keyword_score = evaluate_keyword_match(ref_keywords, resp_keywords)

        # 가중치 적용 최종 점수 (60% 키워드, 40% 유사도)
        final_score = (0.6 * keyword_score + 0.4 * similarity_score) * 100

        return {
            "점수": round(final_score, 2),
            "세부사항": {
                "키워드_일치": round(keyword_score * 100, 2),
                "의미_유사도": round(similarity_score * 100, 2),
                "기준_키워드": ref_keywords,
                "응답_키워드": resp_keywords
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run with: uvicorn main:app --reload