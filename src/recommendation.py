from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from utils import parse_json_safely
from persona import llm

# 영화 추천 템플릿
recommendation_template = """
페르소나:
{persona}

영화 후보 리스트:
{movie_candidates}

이미 시청한 영화 기록:
{watched_movies}

추가 정보:
- 사용자의 영화 감상 목적: {purpose}
- 현재 감정 상태: {current_mood}
- 시청 환경: {viewing_environment}
- 선호하는 영화 길이: {preferred_duration}
- 자막 선호 여부: {subtitle_preference}
- 평점/리뷰의 중요도: {rating_importance}
- 최근 시청 트렌드: {recent_viewing_trend}
- 요청 시각: {request_time}

### 영화 추천 작업
위 정보를 바탕으로 다음 작업을 수행해주세요:
1. 페르소나와 추가 정보에 적합한 영화 6편을 추천하세요. 추천 시 다음 사항을 고려하세요:
   - 각 영화의 장르와 스타일을 다양하게 선택
   - 사용자의 감정적, 지적 니즈 충족
   - 선호하는 영화 요소(플롯, 캐릭터, 시각적 효과 등) 반영
   - 문화적 배경과 언어 선호도 고려
   - 사용자의 영화 선택 기준(상업성 vs 예술성, 신작 vs 고전) 반영
2. 각 영화에 대한 추천 이유를 상세히 설명해주세요. 이때 페르소나의 특성과 어떻게 연결되는지 구체적으로 언급하세요.
3. 영화 후보 리스트에서 4편을 선택하고, 리스트에 없는 2편의 영화를 추가로 추천해주세요. 추가 추천 영화는 실제로 존재하는 영화여야 하며, 페르소나의 취향을 벗어나는 새로운 경험을 제공할 수 있는 작품을 선정하세요.
4. 사용자의 감정 상태, 시청 환경, 선호하는 영화 길이, 자막 선호 여부 등을 고려하여 추천을 조정하세요. 정보가 주어지지 않은 경우, 페르소나를 바탕으로 합리적인 추론을 해주세요.
5. 요청 시각과 최근 시청 트렌드를 고려하여 추천을 조정하세요. 시간대나 계절에 따라 적합한 영화를 선정하고, 사용자의 최근 관심사를 반영하세요.
6. 총 6편의 영화를 추천하고, 각 영화에 대한 이유를 1~5번까지의 작업을 종합적으로 고려해 설명하세요.

결과를 다음 JSON 형식으로 반환해주세요:
{{
    "recommendations": [
        {{
            "title": "영화 제목",
            "reason": "추천 이유"
        }},
        ...
    ],
    "persona_comment": "페르소나의 영화 취향 한 줄 설명"
}}
"""

recommendation_prompt = PromptTemplate(
    input_variables=[
        "persona",
        "movie_candidates",
        "watched_movies",
        "purpose",
        "current_mood",
        "viewing_environment",
        "preferred_duration",
        "subtitle_preference",
        "rating_importance",
        "recent_viewing_trend",
        "request_time"
    ],
    template=recommendation_template
)

recommendation_chain = LLMChain(llm=llm, prompt=recommendation_prompt)

from datetime import datetime

def recommend_movies(request):
    try:
        # 현재 시간을 문자열로 변환
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        recommendations = recommendation_chain.run(
            persona=request.persona,
            movie_candidates=", ".join(request.movie_candidates),
            watched_movies=", ".join(request.watched_movies),
            purpose=request.purpose,
            current_mood=request.current_mood if request.current_mood else "정보 없음",
            viewing_environment=request.viewing_environment,
            preferred_duration=request.preferred_duration if request.preferred_duration else "정보 없음",
            subtitle_preference=request.subtitle_preference if request.subtitle_preference else "정보 없음",
            rating_importance=request.rating_importance if request.rating_importance else "정보 없음",
            recent_viewing_trend=request.recent_viewing_trend if request.recent_viewing_trend else "정보 없음",
            request_time=current_time
        )
        return parse_json_safely(recommendations)
    except Exception as e:
        raise ValueError(f"Error recommending movies: {e}")