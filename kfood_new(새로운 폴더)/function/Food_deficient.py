class FoodLog:
    def __init__(self):
        self.food_log = []

    def record_food(self, food_name, calories, protein, carbs, fat, sodium):
        food_entry = {
            "food_name": food_name,
            "calories": calories,
            "protein": protein,
            "carbs": carbs,
            "fat": fat,
            "sodium": sodium
        }
        self.food_log.append(food_entry)

    def view_food_log(self):
        for entry in self.food_log:
            print(f"음식: {entry['food_name']}")
            print(f"칼로리: {entry['calories']} kcal")
            print(f"단백질: {entry['protein']} g")
            print(f"탄수화물: {entry['carbs']} g")
            print(f"식이섬유: {entry['fibre']} g")
            print("-" * 30)

    def calculate_shortage(self, daily_intake):
        shortage = {
            "calories": daily_intake["calories"],
            "protein": daily_intake["protein"],
            "carbs": daily_intake["carbs"],
            "fat": daily_intake["fat"],
            "sodium": daily_intake["sodium"]
        }

        for entry in self.food_log:
            for nutrient in shortage.keys():
                shortage[nutrient] -= entry[nutrient]

        return shortage

def main():
    food_logger = FoodLog()

    while True:
        print("1. 음식 기록")
        print("2. 음식 기록 보기")
        print("3. 계산하기")
        print("4. 종료")
        choice = input("선택: ")

        if choice == '1':
            food_name = input("음식 이름: ")
            calories = float(input("칼로리: "))
            protein = float(input("단백질 (g): "))
            carbs = float(input("탄수화물 (g): "))
            fibre = float(input("식이섬유 (g): "))
            food_logger.record_food(food_name, calories, protein, carbs, fat, sodium)
            print("음식이 기록되었습니다.")
        elif choice == '2':
            food_logger.view_food_log()
        elif choice == '3':
            daily_intake = {
                "calories": float(input("하루 권장 칼로리 (kcal): ")),
                "protein": float(input("하루 권장 단백질 (g): ")),
                "carbs": float(input("하루 권장 탄수화물 (g): ")),
                "fibre": float(input("하루 권장 식이섬유 (g): "))
            }
            shortage = food_logger.calculate_shortage(daily_intake)
            print("부족한 영양소:")
            for nutrient, value in shortage.items():
                if value < 0:
                    print(f"{nutrient.capitalize()}: {abs(value)}")
            print("-" * 30)
        elif choice == '4':
            break
        else:
            print("잘못된 선택입니다. 다시 시도하세요.")

if __name__ == "__main__":
    main()
