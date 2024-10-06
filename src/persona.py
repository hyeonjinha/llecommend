from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from utils import parse_json_safely
from langchain.llms import Ollama

llm = Ollama(model="eeve:latest")
# 페르소나 생성 템플릿
persona_template = """
당신은 영화 추천을 위한 사용자 페르소나를 만드는 전문가입니다. 주어진 정보를 바탕으로 JSON 형식의 상세한 페르소나를 생성해주세요.

사용자 정보:
- 나이: {age}
- 성별: {gender}
- 직업: {job}
- 취미와 관심사: {hobbies}
- 성격 특성: {personality_traits}
- 선호하는 영화 장르: {preferred_genres}
- 좋아하는 영화 3-5편: {favorite_movies}
- 싫어하는 영화 유형이나 요소: {disliked_elements}
- 영화 감상 목적: {purpose}
- 평소 영화 시청 빈도: {viewing_frequency}
- 선호하는 감독이나 배우: {favorite_creators}
- 주로 영화를 시청하는 환경: {viewing_environment}

제공된 정보 중 일부가 비어있을 수 있습니다. 빈 값이 있으면 해당 정보를 무시하거나 그에 맞는 추론을 해주세요.

### 페르소나 생성 작업:
1. 주어진 정보를 바탕으로 사용자의 성격, 취미 및 영화 선호도를 추론하세요.
2. 사용자의 나이, 직업, 생활 패턴이 영화 선택에 미치는 영향을 고려하세요.
3. 감정적, 지적 니즈에 따른 영화 선호도를 설명하세요.
4. 특정 장르, 테마, 영화적 요소에 대한 호불호를 추론하세요.
5. 영화 시청 습관과 환경을 고려한 추천 전략을 제시하세요.
6. 문화적 배경과 언어 선호도를 추론하여 반영하세요.
7. 영화 선택 기준(평점, 리뷰, 상업성 vs 예술성 등)을 추정하세요.
8. 제공된 정보가 부족한 경우, 적절한 추론을 통해 완성된 페르소나 설명을 제공하세요.

결과는 다음 JSON 형식으로 제공해주세요:

{{
  "persona": "페르소나에 대한 설명"
}}

상세하고 일관된 페르소나를 생성해주세요.
"""

persona_prompt = PromptTemplate(
    input_variables=[
        "age",
        "gender",
        "job",
        "hobbies",
        "personality_traits",
        "preferred_genres",
        "favorite_movies",
        "disliked_elements",
        "purpose",
        "viewing_frequency",
        "favorite_creators",
        "viewing_environment"
    ],
    template=persona_template
)

persona_chain = LLMChain(llm=llm, prompt=persona_prompt)

def generate_persona(user_info):
    try:
        persona = persona_chain.run(
            age=user_info.age if user_info.age else "정보 없음",
            gender=user_info.gender if user_info.gender else "정보 없음",
            job=user_info.job if user_info.job else "정보 없음",
            hobbies=", ".join(user_info.hobbies) if user_info.hobbies else "정보 없음",
            personality_traits=", ".join(user_info.personality_traits) if user_info.personality_traits else "정보 없음",
            preferred_genres=", ".join(user_info.preferred_genres) if user_info.preferred_genres else "정보 없음",
            favorite_movies=", ".join(user_info.favorite_movies) if user_info.favorite_movies else "정보 없음",
            disliked_elements=", ".join(user_info.disliked_elements) if user_info.disliked_elements else "정보 없음",
            purpose=user_info.purpose if user_info.purpose else "정보 없음",
            viewing_frequency=user_info.viewing_frequency if user_info.viewing_frequency else "정보 없음",
            favorite_creators=", ".join(user_info.favorite_creators) if user_info.favorite_creators else "정보 없음",
            viewing_environment=user_info.viewing_environment if user_info.viewing_environment else "정보 없음"
        )
        print(persona)
        return parse_json_safely(persona)
    except Exception as e:
        raise ValueError(f"Error generating persona: {e}")

    
# 페르소나 생성 끝
    
# 페르소나 업데이트 템플릿
update_persona_template = """
당신은 영화 추천을 위한 사용자 페르소나를 업데이트하는 전문가입니다. 아래의 기존 페르소나와 새로운 정보를 바탕으로 업데이트된 페르소나를 생성해주세요.

기존 페르소나:
{existing_persona}

새로운 정보:
{user_input}
(나이, 성별, 직업, 취미와 관심사, 성격 특성, 이미 시청한 영화, 선호하는 감독이나 배우, 영화 감상 목적 등을 포함할 수 있습니다.)

제공된 정보 중 일부가 비어있을 수 있습니다. 빈 값이 있으면 해당 정보를 무시하거나 그에 맞는 추론을 해주세요.
기존 페르소나는 다음과 같은 과정을 통해 생성되었습니다.
### 페르소나 생성 작업:
1. 주어진 정보를 바탕으로 사용자의 성격, 취미 및 영화 선호도를 추론하세요.
2. 사용자의 나이, 직업, 생활 패턴이 영화 선택에 미치는 영향을 고려하세요.
3. 감정적, 지적 니즈에 따른 영화 선호도를 설명하세요.
4. 특정 장르, 테마, 영화적 요소에 대한 호불호를 추론하세요.
5. 영화 시청 습관과 환경을 고려한 추천 전략을 제시하세요.
6. 문화적 배경과 언어 선호도를 추론하여 반영하세요.
7. 영화 선택 기준(평점, 리뷰, 상업성 vs 예술성 등)을 추정하세요.
8. 제공된 정보가 부족한 경우, 적절한 추론을 통해 완성된 페르소나 설명을 제공하세요.

위 정보를 바탕으로 다음 작업을 수행해주세요:

### 페르소나 업데이트 작업:
1. 기존 페르소나의 특성과 사용자가 제공한 새로운 정보를 바탕으로 업데이트된 페르소나를 생성하세요.
2. 새로운 정보가 기존 페르소나에 미치는 영향을 고려하여, 기존의 특성을 조정하거나 추가하세요.
3. 업데이트된 페르소나 설명은 JSON 형식으로 제공되어야 하며, "updated_persona": "업데이트된 페르소나"의 형식으로 반환해야 합니다.

결과는 JSON 형식으로 반환해주세요:
{{
    "updated_persona": "생성된 업데이트된 페르소나"
}}
"""

update_persona_prompt = PromptTemplate(
    input_variables=["existing_persona", "user_input"],
    template=update_persona_template
)

update_persona_chain = LLMChain(llm=llm, prompt=update_persona_prompt)

# 페르소나 업데이트
def update_persona(existing_persona, user_input):
    try:
        updated_persona = update_persona_chain.run(
            existing_persona=existing_persona,
            user_input=user_input
        )
        return parse_json_safely(updated_persona)
    except Exception as e:
        raise ValueError(f"Error updating persona: {e}")