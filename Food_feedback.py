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
            print(f"지방: {entry['fat']} g")
            print(f"나트륨: {entry['sodium']} mg")
            print("-" * 30)

    def calculate_excess(self, daily_intake):
        excess = {
            "calories": self.food_log[-1]['calories'] - daily_intake['calories'],
            "protein": self.food_log[-1]['protein'] - daily_intake['protein'],
            "carbs": self.food_log[-1]['carbs'] - daily_intake['carbs'],
            "fat": self.food_log[-1]['fat'] - daily_intake['fat'],
            "sodium": self.food_log[-1]['sodium'] - daily_intake['sodium']
        }
        return excess

def provide_info_and_recommendations(excess):
    if excess['calories'] > 0:
        print("과다한 칼로리 섭취로 인해 다음과 같은 증상이 나타날 수 있습니다:")
        print("- 체중 증가")
        print("- 혈당 상슨")
        print("- 고혈압")
        print("조치를 취하기 위해 다음을 고려하세요:")
        print("- 칼로리 제한을 설정")
        print("- 식사 크기를 줄이기")
        print("대체 음식 추천: 식사의 일부를 식사 대용 스무디로 대체")
    
    if excess['protein'] > 0:
        print("과다한 단백질 섭취로 인해 다음과 같은 증상이 나타날 수 있습니다:")
        print("- 신장 부담")
        print("- 골다공증 가능성 증가")
        print("조치를 취하기 위해 다음을 고려하세요:")
        print("- 단백질 섭취를 조절")
        print("- 식사에서 식물성 단백질을 추가")
        print("대체 음식 추천: 채소 기반 단백질 식사")

    # 다른 영양소에 대한 증상 정보 및 대체 음식 정보 추가

def main():
    food_logger = FoodLog()

    while True:
        print("1. 음식 기록")
        print("2. 음식 기록 보기")
        print("3. 평가 및 조치")
        print("4. 종료")
        choice = input("선택: ")

        if choice == '1':
            food_name = input("음식 이름: ")
            calories = float(input("칼로리: "))
            protein = float(input("단백질 (g): "))
            carbs = float(input("탄수화물 (g): "))
            fat = float(input("지방 (g): "))
            sodium = float(input("나트륨 (mg): "))
            food_logger.record_food(food_name, calories, protein, carbs, fat, sodium)
            print("음식이 기록되었습니다.")
        elif choice == '2':
            food_logger.view_food_log()
        elif choice == '3':
            daily_intake = {
                "calories": float(input("하루 권장 칼로리 (kcal): ")),
                "protein": float(input("하루 권장 단백질 (g): ")),
                "carbs": float(input("하루 권장 탄수화물 (g): ")),
                "fat": float(input("하루 권장 지방 (g): ")),
                "sodium": float(input("하루 권장 나트륨 (mg): "))
            }
            excess = food_logger.calculate_excess(daily_intake)
            provide_info_and_recommendations(excess)
        elif choice == '4':
            break
        else:
            print("잘못된 선택입니다. 다시 시도하세요.")

if __name__ == "__main__":
    main()
