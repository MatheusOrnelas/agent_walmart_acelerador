import io
import logging
from PIL import Image
from langchain_core.runnables.graph import GraphRunnable

logger = logging.getLogger(__name__)

def save_graph_figure(graph_runnable, output_path: str):
    """
    Saves a visualization of a compiled LangGraph to a file.

    This function generates a PNG image from the graph's structure
    and saves it to the specified path.

    Args:
        graph_runnable: The compiled LangGraph object (result of graph.compile()).
        output_path (str): The path to save the PNG file (e.g., 'graph.png').
    """
    try:
        # Get the PNG image data as bytes from the runnable graph
        png_bytes = graph_runnable.get_graph().draw_mermaid_png()

        # Use Pillow to open the image from the bytes and save it
        image = Image.open(io.BytesIO(png_bytes))
        image.save(output_path)
        logger.info(f"✅ Graph figure saved successfully to: {output_path}")

    except Exception as e:
        logger.error(f"❌ An error occurred while saving the graph figure: {e}")

