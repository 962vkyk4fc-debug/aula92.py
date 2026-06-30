import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

# Configuração da página do Streamlit
st.set_page_config(page_title="Detector de Objetos IA", layout="centered")

st.title("📸 Identificação Inteligente de Imagens")
st.write("Faça o upload de uma imagem para detectar e categorizar os objetos presentes de forma automática.")

# Cache do modelo para evitar recarregamento a cada interação
@st.cache_resource
def load_model():
    # Carrega a versão Nano (leve, ideal para CPU)
    return YOLO("yolov8n.pt")

try:
    model = load_model()
except Exception as e:
    st.error(f"Erro ao carregar o modelo YOLOv8: {e}")
    st.stop()

# Componente de upload de imagem
uploaded_file = st.file_uploader("Escolha uma imagem...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Converter o arquivo enviado para uma imagem PIL
    image = Image.open(uploaded_file)
    
    # Layout de colunas para comparar o antes e depois
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Imagem Original")
        st.image(image, use_container_width=True)
        
    with st.spinner("Processando imagem com YOLOv8..."):
        # Converter PIL para array numpy (padrão OpenCV/YOLO)
        img_array = np.array(image)
        
        # Executar a inferência do modelo
        results = model(img_array)
        
        # O YOLO retorna os resultados anotados em formato BGR (OpenCV)
        # Extraímos a primeira imagem do lote de resultados
        res_plotted = results[0].plot()
        
        # Converter de BGR de volta para RGB para exibição correta no Streamlit
        res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
        
    with col2:
        st.subheader("Objetos Identificados")
        st.image(res_rgb, use_container_width=True)

    # Exibir métricas e classes encontradas abaixo das imagens
    st.markdown("---")
    st.subheader("📋 Relatório de Detecção")
    
    boxes = results[0].boxes
    if len(boxes) == 0:
        st.info("Nenhum objeto conhecido foi detectado na imagem.")
    else:
        # Contagem dos objetos detectados
        detected_objects = {}
        for box in boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            confidence = float(box.conf[0]) * 100
            
            if class_name not in detected_objects:
                detected_objects[class_name] = []
            detected_objects[class_name].append(confidence)
        
        # Exibir em formato de lista limpa
        for obj, confidences in detected_objects.items():
            qtd = len(confidences)
            conf_media = sum(confidences) / qtd
            st.write(f"• **{obj.capitalize()}**: {qtd} encontrado(s) (Confiança média: {conf_media:.1f}%)")