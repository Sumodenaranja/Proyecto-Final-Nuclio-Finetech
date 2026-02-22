import nbformat

files = [
    r"C:\Users\marcl\pythonProjects\Proyecto-Final-Nuclio-Finetech\Grupo_1_Analisis_Fintech.ipynb",
    r"C:\Users\marcl\pythonProjects\Proyecto-Final-Nuclio-Finetech\Grupo_1_Analisis_Fintech.ipynb"
]

for f in files:
    try:
        nbformat.read(f, as_version=4)
        print(f"{f} is valid")
    except Exception as e:
        print(f"{f} is INVALID: {e}")