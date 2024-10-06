from fastapi import FastAPI, HTTPException
import uvicorn
from persona import generate_persona, update_persona
from recommendation import recommend_movies
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI()

class UserInfo(BaseModel):
    age: int
    gender: str
    job: str
    hobbies: List[str]
    personality_traits: List[str]
    preferred_genres: List[str]
    favorite_movies: List[str]
    disliked_elements: List[str]
    purpose: str
    viewing_frequency: str
    favorite_creators: List[str]
    viewing_environment: str

class MovieRecommendationRequest(BaseModel):
    persona: str
    movie_candidates: List[str]
    watched_movies: List[str]
    purpose: str
    current_mood: Optional[str]
    viewing_environment: str
    preferred_duration: Optional[str]
    subtitle_preference: Optional[str]
    rating_importance: Optional[str]
    recent_viewing_trend: Optional[str]
    request_time: str

@app.post("/generate_persona")
async def generate_persona_endpoint(user_info: UserInfo):
    try:
        persona = generate_persona(user_info)
        return persona
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/recommend_movies")
async def recommend_movies_endpoint(request: MovieRecommendationRequest):
    try:
        recommendations = recommend_movies(request)
        return recommendations
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/update_persona")
async def update_persona_endpoint(existing_persona: str, user_input: str):
    try:
        updated_persona = update_persona(existing_persona, user_input)
        return updated_persona
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)