Here’s a sample `README.md` for your crowd analysis dashboard using Streamlit and YOLO:

```markdown
# Dashboard de Análise de Multidões

Este aplicativo foi desenvolvido para realizar a análise de multidões a partir de vídeos, utilizando o modelo YOLO (You Only Look Once) para detecção de pessoas e gerar um heatmap com informações sobre a densidade de pessoas ao longo do tempo.

## Funcionalidades

- **Detecção de Pessoas**: O aplicativo utiliza o modelo YOLOv8 para identificar pessoas em vídeos.
- **Heatmap de Densidade**: Gera um heatmap visual que mostra a densidade de pessoas em diferentes áreas do vídeo.
- **Gráficos Interativos**: Apresenta gráficos que mostram o número de pessoas detectadas e a densidade média ao longo do tempo.
- **Download de Vídeo Processado**: Permite o download do vídeo processado com informações de densidade e detecção de pessoas.

## Tecnologias Utilizadas

- **Streamlit**: Para a interface do usuário.
- **OpenCV**: Para manipulação de vídeo e processamento de imagens.
- **YOLO (Ultralytics)**: Para detecção de objetos (pessoas).
- **Pandas**: Para manipulação de dados.
- **Matplotlib**: Para visualização de gráficos.

## Requisitos

Para executar este aplicativo, você precisará dos seguintes pacotes Python:

```bash
pip install streamlit opencv-python numpy pandas matplotlib ultralytics
```

## Como Executar o Aplicativo

1. Clone este repositório ou baixe os arquivos.
2. Navegue até o diretório do projeto.
3. Execute o aplicativo Streamlit com o seguinte comando:

   ```bash
   streamlit run app.py
   ```

4. Acesse o aplicativo no seu navegador no endereço `http://localhost:8501`.

## Como Usar

1. **Upload do Vídeo**: Carregue um vídeo no formato MP4, AVI, MOV ou MKV através da barra lateral.
2. **Iniciar Processamento**: Clique no botão "Iniciar Processamento" para começar a análise do vídeo.
3. **Pausar/Continuar**: Use os botões para pausar ou continuar o processamento conforme necessário.
4. **Resultados**: Após a conclusão do processamento, os gráficos serão exibidos na tela e você poderá baixar o vídeo processado.

## Contribuição

Sinta-se à vontade para contribuir com melhorias ou correções. Envie um pull request ou abra uma issue.

## Licença

Este projeto está licenciado sob a [MIT License](LICENSE).

## Contato

Para mais informações ou perguntas, entre em contato:

- **LinkedIn**: [Vinicius Bertotto](https://www.linkedin.com/in/vinicius-bertotto/)
- **GitHub**: [vbertotto](https://github.com/vbertotto)
- **Website**: [bertotto.online](https://bertotto.online/)
```
