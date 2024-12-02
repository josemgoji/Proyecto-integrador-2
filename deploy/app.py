import gradio as gr
from pipelineFinal import pipeline_final

demo = gr.Interface(
   fn=pipeline_final,
   inputs=[
       gr.Dropdown(["Si", "No"], label="Deseas hacer una prediccion justo luego del tiempo del train"),  # Uncomment this line to add audio input
       gr.Slider(0, 24, label="Choose a number"),
       gr.File(label="Sube el archivo de tain en csv. (Solo si elegiste NO)"),
       gr.File(label="Sube el archivo de client en csv. (Solo si elegiste NO)"),
       gr.File(label="Sube el archivo de histroical_weather en csv. (Solo si elegiste NO)"),
       gr.File(label="Sube el archivo de electricity_prices en csv. (Solo si elegiste NO)"),
       gr.File(label="Sube el archivo de gas_prices en csv. (Solo si elegiste NO)")

   ],
   outputs=[gr.Plot(), gr.DataFrame()]
)
demo.launch()

