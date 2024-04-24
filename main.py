from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib

# Load the trained model
model_path = 'model/food_model.h5'
loaded_model = joblib.load(model_path)

# Define the input data schema
class FoodItem(BaseModel):
    Glycemic_Index: float
    Calories: float
    Carbohydrates: float
    Protein: float
    Fat: float
    Sodium_Content: int
    Potassium_Content: int
    Magnesium_Content: int
    Calcium_Content: int
    Fiber_Content: float

# Create an instance of the FastAPI class
app = FastAPI()

# Define a route to make predictions
@app.post("/predict/")
async def predict_food_item(item: FoodItem):
    # Convert input data into a format expected by the model
    new_data = [[item.Glycemic_Index, item.Calories, item.Carbohydrates, item.Protein, item.Fat,
                 item.Sodium_Content, item.Potassium_Content, item.Magnesium_Content,
                 item.Calcium_Content, item.Fiber_Content]]
    
    # Make predictions using the loaded model
    prediction = loaded_model.predict(new_data)
    
    # Return the prediction
    return {"prediction": prediction[0]}

# Run the FastAPI application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
