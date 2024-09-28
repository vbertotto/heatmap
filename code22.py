import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO, solutions
import pandas as pd
import matplotlib.pyplot as plt
import tempfile
import os
from io import BytesIO
import time
import threading

# Configuração da página
st.set_page_config(page_title="Análise de Multidões", layout="wide")

# Título do Aplicativo
st.title("Dashboard de Análise de Multidões")

# Função para carregar o modelo YOLO com cache para otimizar desempenho
@st.cache_resource
def load_model():
    model = YOLO("yolov8n.pt")  # Certifique-se de que o arquivo do modelo está disponível
    return model

# Carrega o modelo YOLO
model = load_model()

# Função para inicializar a captura de vídeo
def initialize_video_capture(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception("Erro ao ler o arquivo de vídeo")
    return cap

# Função para inicializar o heatmap
def initialize_heatmap(model):
    return solutions.Heatmap(
        names=model.names,  # Adicionado o argumento 'names'
        colormap=cv2.COLORMAP_PARULA,
        view_img=False,  
        shape="circle"
    )

# Função para inicializar o gravador de vídeo
def initialize_video_writer(output_path, fps, frame_size):
    return cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, frame_size)

# Função para calcular a densidade
def calculate_density(grid_size, w, h, person_boxes):
    regions = np.zeros((grid_size, grid_size))  # Inicializa a grade para contagem de pessoas
    for box in person_boxes:
        center_x, center_y, _, _ = box
        grid_x = min(int((center_x / w) * grid_size), grid_size - 1)
        grid_y = min(int((center_y / h) * grid_size), grid_size - 1)
        regions[grid_y, grid_x] += 1
    return regions, np.mean(regions)

# Classe para gerenciar o processamento do vídeo
class VideoProcessor:
    def __init__(self, video_path, output_path, model, heatmap_obj, grid_size, w, h, fps):
        self.video_path = video_path
        self.output_path = output_path
        self.model = model
        self.heatmap_obj = heatmap_obj
        self.grid_size = grid_size
        self.w = w
        self.h = h
        self.fps = fps
        self.data_log = []
        self.prev_density = 0
        self.prev_num_persons = 0
        self.paused = False
        self.stop_processing = False
        self.thread = None
        self.frame_count = 0

    def start_processing(self):
        if self.thread is None:
            self.thread = threading.Thread(target=self.process_video)
            self.thread.start()

    def pause_processing(self):
        self.paused = True

    def resume_processing(self):
        self.paused = False

    def stop_processing_thread(self):
        self.stop_processing = True
        if self.thread is not None:
            self.thread.join()

    def process_video(self):
        cap = initialize_video_capture(self.video_path)
        video_writer = initialize_video_writer(self.output_path, self.fps, (self.w, self.h))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for idx in range(total_frames):
            if self.stop_processing:
                break
            while self.paused:
                time.sleep(0.5)
                if self.stop_processing:
                    break
            success, im0 = cap.read()
            if not success:
                break

            im0, self.prev_density, self.prev_num_persons = process_frame(
                im0, self.model, self.heatmap_obj, self.grid_size, self.w, self.h, self.data_log, self.prev_density, self.prev_num_persons, cap, self.fps
            )

            video_writer.write(im0)

            self.frame_count = idx + 1

            # Simula um tempo de processamento
            time.sleep(0.01)

        cap.release()
        video_writer.release()

# Função para processar cada frame
def process_frame(im0, model, heatmap_obj, grid_size, w, h, data_log, prev_density, prev_num_persons, cap, fps):
    tracks = model.track(im0, persist=True, show=False)
    if tracks and tracks[0].boxes and tracks[0].boxes.id is not None:
        im0 = heatmap_obj.generate_heatmap(im0, tracks)
        person_boxes = tracks[0].boxes.xywh.cpu().numpy()
        person_ids = tracks[0].boxes.id.cpu().numpy()
        num_persons = len(person_ids)
        regions, density = calculate_density(grid_size, w, h, person_boxes)

        # Calcula as mudanças temporais
        density_change = density - prev_density
        num_persons_change = num_persons - prev_num_persons

        # Registra os dados da frame para análise
        data_log.append({
            "frame": int(cap.get(cv2.CAP_PROP_POS_FRAMES)),
            "time_sec": int(cap.get(cv2.CAP_PROP_POS_FRAMES)) / fps,
            "num_persons": num_persons,
            "average_density": density,
            "density_change": density_change,
            "num_persons_change": num_persons_change
        })

        # Desenha a grade e a densidade no frame do vídeo
        for i in range(grid_size):
            for j in range(grid_size):
                cv2.putText(im0, str(int(regions[j, i])), 
                            (i * (w // grid_size), j * (h // grid_size)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return im0, density, num_persons
    return im0, prev_density, prev_num_persons

# Função para plotar os gráficos
def plot_graphs(df):
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(df["time_sec"], df["num_persons"], label="Número de Pessoas")
    ax1.plot(df["time_sec"], df["num_persons"].rolling(window=30).mean(), label="Tendência - Número de Pessoas", linestyle="--")
    ax1.set_xlabel("Tempo (segundos)")
    ax1.set_ylabel("Número de Pessoas")
    ax1.set_title("Número de Pessoas Detectadas ao Longo do Tempo")
    ax1.legend()
    ax1.grid(True)

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.plot(df["time_sec"], df["average_density"], label="Densidade Média")
    ax2.plot(df["time_sec"], df["average_density"].rolling(window=30).mean(), label="Tendência - Densidade Média", linestyle="--")
    ax2.set_xlabel("Tempo (segundos)")
    ax2.set_ylabel("Densidade Média")
    ax2.set_title("Densidade Média ao Longo do Tempo")
    ax2.legend()
    ax2.grid(True)

    return fig1, fig2

# Função para remover arquivos temporários com tentativas
def safe_remove(file_path, retries=10, delay=0.5):
    for attempt in range(retries):
        try:
            os.remove(file_path)
            break
        except PermissionError:
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                st.warning(f"Não foi possível remover o arquivo temporário: {file_path}")

# Função para adicionar ícones de redes sociais na barra lateral
def add_social_links():
    st.sidebar.markdown("---")
    st.sidebar.header("Siga-me nas Redes Sociais")
    # Substitua os links abaixo pelos seus URLs reais
    linkedin_url = "https://www.linkedin.com/in/vinicius-bertotto/"
    github_url = "https://github.com/vbertotto"
    website_url = "https://bertotto.online/"

    # Usando emojis como ícones
    st.sidebar.markdown(f"""
    <a href="{website_url}" target="_blank">🌐 Website</a>  
    <br>
    <a href="{linkedin_url}" target="_blank">🔗 LinkedIn</a>  
    <br>
    <a href="{github_url}" target="_blank">🐙 GitHub</a>  
    """, unsafe_allow_html=True)

# Adiciona os links de redes sociais
add_social_links()

# Interface de Upload de Vídeo
st.sidebar.header("Upload do Vídeo")
uploaded_file = st.sidebar.file_uploader("Escolha um arquivo de vídeo", type=["mp4", "avi", "mov", "mkv"])

# Initialize session_state variables
if 'processor' not in st.session_state:
    st.session_state['processor'] = None
if 'data_log' not in st.session_state:
    st.session_state['data_log'] = []
if 'df' not in st.session_state:
    st.session_state['df'] = None

if uploaded_file is not None:
    # Salva o arquivo de vídeo temporariamente
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    tfile.close()  # Fechar o arquivo temporário após a escrita
    video_path = tfile.name

    # Inicializa a captura de vídeo
    try:
        cap = initialize_video_capture(video_path)
    except Exception as e:
        st.error(f"Erro ao abrir o vídeo: {e}")
        safe_remove(video_path)  # Tenta remover mesmo em caso de erro
        st.stop()

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    grid_size = 10  # Tamanho da grade para dividir o frame

    cap.release()  # Libera imediatamente após obter informações necessárias

    # Inicializa o heatmap e o gravador de vídeo
    heatmap_obj = initialize_heatmap(model)
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video_file:
        output_path = temp_video_file.name
    video_writer = initialize_video_writer(output_path, fps, (w, h))
    video_writer.release()  # Libera imediatamente para evitar conflitos

    # Registro de dados e valores iniciais
    data_log = []
    prev_density = 0
    prev_num_persons = 0

    # Barra de progresso
    frame_count = int(cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FRAME_COUNT))
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Initialize the VideoProcessor in session_state
    if st.session_state['processor'] is None:
        st.session_state['processor'] = VideoProcessor(
            video_path, output_path, model, heatmap_obj, grid_size, w, h, fps
        )

    processor = st.session_state['processor']

    # Controles de processamento
    if not processor.thread or not processor.thread.is_alive():
        start_button = st.sidebar.button("Iniciar Processamento")
    else:
        start_button = False

    if start_button:
        processor.start_processing()
        st.sidebar.success("Processamento iniciado.")

    # Botão de pausar/resumir
    if processor.thread and processor.thread.is_alive():
        if processor.paused:
            resume_button = st.sidebar.button("Continuar Processamento")
            if resume_button:
                processor.resume_processing()
                st.sidebar.success("Processamento continuado.")
        else:
            pause_button = st.sidebar.button("Pausar Processamento")
            if pause_button:
                processor.pause_processing()
                st.sidebar.warning("Processamento pausado.")

    # Atualiza o progresso
    if processor.thread and processor.thread.is_alive():
        progress = processor.frame_count / frame_count
        progress_bar.progress(progress)
        status_text.text(f"Processando frame {processor.frame_count} de {frame_count}")
    elif processor.thread and not processor.thread.is_alive():
        progress_bar.progress(1.0)
        status_text.text("Processamento concluído!")

        # Finaliza o processamento e fecha os arquivos
        processor.stop_processing_thread()

        # Carrega os dados do log em um DataFrame
        df = pd.DataFrame(processor.data_log)
        st.session_state['df'] = df

        # Exibe os gráficos
        st.header("Resultados da Análise")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Número de Pessoas ao Longo do Tempo")
            fig1, _ = plot_graphs(df)
            st.pyplot(fig1)

        with col2:
            st.subheader("Densidade Média ao Longo do Tempo")
            _, fig2 = plot_graphs(df)
            st.pyplot(fig2)

        # Opção para baixar o vídeo processado
        st.header("Download do Vídeo Processado")
        try:
            with open(processor.output_path, 'rb') as video_file:
                video_bytes = video_file.read()
            st.download_button(
                label="Baixar Vídeo Processado",
                data=video_bytes,
                file_name='mapa_de_calor.mp4',
                mime='video/mp4',
            )
        except Exception as e:
            st.error(f"Erro ao preparar o vídeo para download: {e}")

        # Opção para baixar o log CSV
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Baixar Log de Análise (CSV)",
            data=csv,
            file_name='crowd_analysis_log.csv',
            mime='text/csv',
        )

        # Tentativa de remover os arquivos temporários com retries
        safe_remove(video_path)
        safe_remove(processor.output_path)

else:
    st.info("Por favor, faça o upload de um arquivo de vídeo para começar a análise.")
