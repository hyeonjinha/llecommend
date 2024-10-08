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
- 유저가 제공한 추가 정보: {user_input}

제공된 정보 중 일부가 비어있을 수 있습니다. 빈 값이 있으면 해당 정보를 무시하거나 그에 맞는 추론을 해주세요.

### 페르소나 생성 작업:
1. 사용자의 성격, 취미 및 영화 선호도를 주어진 정보에 맞춰 추론하세요.
2. 감정 상태와 영화 감상 목적이 영화 선택에 미치는 영향을 반영하여 페르소나를 작성하세요.
3. 나이, 직업, 생활 패턴을 고려하여 영화 선택 기준(예: 상업성 vs 예술성, 신작 vs 고전)을 추정하세요.
4. 특정 장르, 테마, 영화적 요소에 대한 호불호를 추론하여 정리하세요.
5. 문화적 배경과 언어 선호도를 반영하여 영화를 추천할 수 있도록 페르소나를 작성하세요.
6. 제공된 정보가 부족한 경우, 적절한 추론을 통해 완성된 페르소나 설명을 제공하세요.

결과는 다음 JSON 형식으로 제공해주세요:

{
  "persona": "페르소나에 대한 설명"
}

상세하고 일관된 페르소나를 생성해주세요.
"""

persona_prompt = PromptTemplate(
    input_variables=[
        "age",
        "gender",
        "job",
        "user_input",
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
            user_input=user_info.user_input if user_info.user_input else "정보 없음"
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

### 페르소나 업데이트 작업:
1. 새로운 정보에 따라 기존 페르소나의 성격, 취미, 영화 선호도를 업데이트하세요.
2. user_input에 추가된 정보가 기존 페르소나에 미치는 영향을 고려하여 적절히 조정하거나 추가하세요.
3. 업데이트된 페르소나 설명은 JSON 형식으로 제공되어야 하며, "updated_persona": "업데이트된 페르소나"의 형식으로 반환해야 합니다.

결과는 JSON 형식으로 반환해주세요:
{
    "updated_persona": "생성된 업데이트된 페르소나"
}
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