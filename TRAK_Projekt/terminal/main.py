import random
import string
from pywavefront import Wavefront

# Pusta funkcja readMap - do zastąpienia implementacją
def readMap(rendering_choice):
    print(f"Wczytano mapę dla renderingu {rendering_choice}.")


def load_obj_file(file_path):
  try:
      mesh = Wavefront(file_path)
      return mesh
  except Exception as e:
      print(f"Error loading OBJ file: {e}")
      return None

# Funkcja obsługująca wczytywanie informacji z pliku obj
def loadInformationFromFile():
    obj_file_path = "file.obj"
    loaded_mesh = load_obj_file(obj_file_path)
  
    if loaded_mesh:
        print(f"Wczytano informacje z pliku obj: {obj_file_path}")
    else:
        print("Problem z plikiem obj.")


# Funkcja obsługująca wybór z listy
def chooseFromList():
    renderings_list = ["rendering 1", "rendering 2", "rendering 3"]

    print("Wybierz rendering z listy:")
    for idx, rendering in enumerate(renderings_list, start=1):
        print(f"{idx}. {rendering}")

    try:
        user_choice = int(input("Twój wybór (1-3): "))
        if 1 <= user_choice <= 3:
            return renderings_list[user_choice - 1]
        else:
            print("Nieprawidłowy wybór.")
            return None
    except ValueError:
        print("Wprowadź poprawną liczbę.")
        return None


# Główna funkcja programu
def main():
    information_loaded = False
    rendering_choice = None

    while True:
        print("\nMenu:")
        print("1. Wczytaj informacje z pliku")
        print("2. Wybierz rendering z listy")
        print("3. Wykonaj funkcję readMap()")
        print("4. Zakończ program")

        choice = input("Wybierz opcję (1-4): ")

        if choice == "1":
            # Wczytaj informacje z pliku
            loadInformationFromFile()
            information_loaded = True
        elif choice == "2":
            # Wybierz rendering z listy
            if information_loaded:
                rendering_choice = chooseFromList()
            else:
                print("Wczytaj najpierw informacje z pliku.")
        elif choice == "3":
            # Wykonaj funkcję readMap()
            if information_loaded and rendering_choice:
                readMap(rendering_choice)
            else:
                print("Wczytaj informacje i dokonaj wyboru z listy przed wykonaniem funkcji.")
        elif choice == "4":
            # Zakończ program
            print("Program zakończony.")
            break
        else:
            print("Nieprawidłowy wybór. Wybierz opcję od 1 do 4.")



if __name__ == "__main__":
    main()