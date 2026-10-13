import sys
from PySide6.QtWidgets import QApplication
from studio import DrillAnnotationStudio

def main():
    app = QApplication(sys.argv)
    window = DrillAnnotationStudio()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
