import pandas as pd

class FoodRecommendation:
    def __init__(self):
        self.food_database = self.update_food_database()
    
    def update_food_database(self):
        # 엑셀 파일에서 음식 정보 불러오기
        excel_file_path = "C:/Users/KITCOOP/kicpython/hansik/kfood_new/xlsx/xlsx_b.xlsx"  # 엑셀 파일 경로
        df = pd.read_excel(excel_file_path)

        # 엑셀 파일에서 음식 정보를 데이터베이스로 변환
        food_database = df.to_dict(orient='records')
        
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
                "칼로리": float(input("부족한 칼로리 (kcal): ")),
                "단백질": float(input("부족한 단백질 (g): ")),
                "탄수화물": float(input("부족한 탄수화물 (g): ")),
                "식이섬유": float(input("식이섬유 (g): "))
            }
            recommended_foods = food_recommendation.recommend_foods(shortage)
            print("부족한 영양소를 보충할 수 있는 음식 추천:")
            for food in recommended_foods:
                print(f"음식: {food['음식 이름']}")
                print(f"{list(shortage.keys())[list(shortage.values()).index(min(shortage.values()))]} 보충용으로 추천됩니다.")
                print("-" * 30)
        elif choice == '2':
            break
        else:
            print("잘못된 선택입니다. 다시 시도하세요.")

if __name__ == "__main__":
    main()
