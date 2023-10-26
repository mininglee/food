import requests
from bs4 import BeautifulSoup

class FoodRecommendation:
    def __init__(self):
        self.food_database = self.update_food_database()
    
    def update_food_database(self):
        # 웹 페이지에서 음식 정보 크롤링
        url = "https://example.com/food_data"  # 원하는 웹 페이지의 URL을 입력
        response = requests.get(url)
        soup = BeautifulSoup(response.content, "html.parser")

        # 웹 페이지에서 음식 정보를 추출하여 업데이트
        food_database = []
        for item in soup.find_all("div", class_="food-item"):
            food_name = item.find("span", class_="food-name").text
            calories = float(item.find("span", class_="calories").text)
            protein = float(item.find("span", class_="protein").text)
            carbs = float(item.find("span", class_="carbs").text)
            fat = float(item.find("span", class_="fat").text)
            food_entry = {
                "food_name": food_name,
                "calories": calories,
                "protein": protein,
                "carbs": carbs,
                "fat": fat
            }
            food_database.append(food_entry)
        
        return food_database

    def recommend_foods(self, shortage):
        recommended_foods = []
        for nutrient, value in shortage.items():
            if value < 0:
                # 부족한 영양소에 대해 적합한 음식 추천
                suitable_foods = [food for food in self.food_database if food[nutrient] > 0]
                if suitable_foods:
                    recommended_food = min(suitable_foods, key=lambda x: x[nutrient])
                    recommended_foods.append(recommended_food)
        return recommended_foods

def main():
    food_recommendation = FoodRecommendation()

    while True:
        print("1. 부족한 영양소 확인 및 음식 추천")
        print("2. 종료")
        choice = input("선택: ")

        if choice == '1':
            shortage = {
                "calories": float(input("부족한 칼로리 (kcal): ")),
                "protein": float(input("부족한 단백질 (g): ")),
                "carbs": float(input("부족한 탄수화물 (g): ")),
                "fat": float(input("부족한 지방 (g): "))
            }
            recommended_foods = food_recommendation.recommend_foods(shortage)
            print("부족한 영양소를 보충할 수 있는 음식 추천:")
            for food in recommended_foods:
                print(f"음식: {food['food_name']}")
                print(f"{list(shortage.keys())[list(shortage.values()).index(min(shortage.values()))]} 보충용으로 추천됩니다.")
                print("-" * 30)
        elif choice == '2':
            break
        else:
            print("잘못된 선택입니다. 다시 시도하세요.")

if __name__ == "__main__":
    main()
