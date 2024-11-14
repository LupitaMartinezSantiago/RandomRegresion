import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import os
import traceback

def ejecutar_notebook(ruta_notebook):
    """
    Ejecuta un archivo Jupyter Notebook y retorna su contenido en formato HTML.
    Solo incluye las celdas que generan alguna salida.
    """
    try:
        # Leer el notebook
        with open(ruta_notebook, 'r') as f:
            notebook = nbformat.read(f, as_version=4)
        
        # Configurar el preprocesador para ejecutar el notebook
        ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
        ep.preprocess(notebook, {'metadata': {'path': os.path.dirname(ruta_notebook)}})

        # Procesar celdas con salida
        output_html = ""
        for index, cell in enumerate(notebook.cells):
            if cell.cell_type == "code" and "outputs" in cell:
                cell_outputs = [
                    output for output in cell.outputs if "text" in output or "data" in output
                ]
                if cell_outputs:  # Solo incluir celdas con salida
                    output_html += f"<h3>Salida de la celda {index + 1}:</h3>"
                    for output in cell_outputs:
                        if "text" in output:
                            output_html += f"<pre>{output['text']}</pre>"
                        elif "data" in output and "text/html" in output["data"]:
                            output_html += output["data"]["text/html"]
        
        # Si no hay salidas, indicar que no se generó contenido
        if not output_html:
            output_html = "<p>No se generaron resultados al ejecutar el notebook.</p>"
        
        return output_html

    except Exception as e:
        # Capturar información detallada sobre el error
        error_trace = traceback.format_exc()
        return (
            f"<pre>Error al ejecutar el notebook: {str(e)}\n\n"
            f"Traza del error:\n{error_trace}</pre>"
        )
