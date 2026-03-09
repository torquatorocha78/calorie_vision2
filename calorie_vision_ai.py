import streamlit as st
from openai import OpenAI, RateLimitError
from PIL import Image
import base64
import io
import os
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import json
import logging
from typing import Dict, Optional
from datetime import datetime

# =============================
# LOGGING
# =============================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================
# CONFIG
# =============================

load_dotenv()

# Validar API Key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("❌ OPENAI_API_KEY não encontrada. Configure a variável de ambiente.")
    st.stop()

client = OpenAI(api_key=api_key)

st.set_page_config(
    page_title="Calorie Vision AI",
    page_icon="🥗",
    layout="centered"
)

# =============================
# CSS
# =============================

st.markdown("""
<style>
.stApp {
    background-color: white;
}

h1, h2, h3 {
    color: #1b8a3b;
}

.result-box {
    background: #e8f5e9;
    padding: 20px;
    border-radius: 10px;
    border-left: 4px solid #1b8a3b;
}

.error-box {
    background: #ffebee;
    padding: 15px;
    border-radius: 8px;
    border-left: 4px solid #c62828;
}

.success-box {
    background: #e8f5e9;
    padding: 15px;
    border-radius: 8px;
    border-left: 4px solid #2e7d32;
}
</style>
""", unsafe_allow_html=True)

# =============================
# DATABASE MANAGER
# =============================

class DatabaseManager:
    """Gerenciador de banco de dados com context manager"""
    
    def __init__(self, db_path: str = "history.db"):
        self.db_path = db_path
        self.init_db()
    
    def init_db(self):
        """Inicializa o banco de dados"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS meals(
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        description TEXT NOT NULL,
                        calories REAL NOT NULL,
                        carbs REAL NOT NULL,
                        protein REAL NOT NULL,
                        fat REAL NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                conn.commit()
                logger.info("Database initialized successfully")
        except sqlite3.Error as e:
            logger.error(f"Database initialization error: {e}")
            raise
    
    def save_meal(self, data: Dict) -> bool:
        """Salva uma refeição no banco de dados"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO meals(description, calories, carbs, protein, fat)
                    VALUES(?, ?, ?, ?, ?)
                """, (
                    data.get("description", ""),
                    float(data.get("calories", 0)),
                    float(data.get("carbs", 0)),
                    float(data.get("protein", 0)),
                    float(data.get("fat", 0))
                ))
                conn.commit()
                logger.info(f"Meal saved: {data.get('description')}")
                return True
        except (sqlite3.Error, ValueError) as e:
            logger.error(f"Error saving meal: {e}")
            return False
    
    def get_history(self) -> pd.DataFrame:
        """Retorna o histórico de refeições"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql(
                    "SELECT id, description, calories, carbs, protein, fat, created_at FROM meals ORDER BY created_at DESC",
                    conn
                )
                return df
        except sqlite3.Error as e:
            logger.error(f"Error loading history: {e}")
            return pd.DataFrame()
    
    def get_summary(self) -> Dict:
        """Retorna resumo de calorias e macros"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT 
                        SUM(calories) as total_calories,
                        SUM(carbs) as total_carbs,
                        SUM(protein) as total_protein,
                        SUM(fat) as total_fat,
                        COUNT(*) as meal_count
                    FROM meals
                """)
                result = cursor.fetchone()
                return {
                    "calories": result[0] or 0,
                    "carbs": result[1] or 0,
                    "protein": result[2] or 0,
                    "fat": result[3] or 0,
                    "meal_count": result[4] or 0
                }
        except sqlite3.Error as e:
            logger.error(f"Error getting summary: {e}")
            return {}


# =============================
# FOOD ANALYZER
# =============================

class FoodAnalyzer:
    """Analisador de comida com IA"""
    
    MAX_RETRIES = 3
    MODEL = "gpt-4o-mini"
    
    def __init__(self, client: OpenAI):
        self.client = client
    
    @staticmethod
    def encode_image(image: Image.Image) -> str:
        """Codifica imagem para base64"""
        try:
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG", quality=85)
            buffered.seek(0)
            return base64.b64encode(buffered.getvalue()).decode()
        except Exception as e:
            logger.error(f"Error encoding image: {e}")
            raise
    
    def analyze_food(self, image: Image.Image) -> Optional[Dict]:
        """
        Analisa foto de comida com retry automático
        """
        base64_image = self.encode_image(image)
        
        prompt = """
Analise esta foto de comida e estime:
- Descrição do prato
- Calorias (em kcal)
- Carboidratos (em gramas)
- Proteínas (em gramas)
- Gorduras (em gramas)

Responda APENAS com JSON válido, sem markdown:
{
    "description": "descrição do prato",
    "calories": 0,
    "carbs": 0,
    "protein": 0,
    "fat": 0
}
"""
        
        for attempt in range(self.MAX_RETRIES):
            try:
                response = self.client.chat.completions.create(
                    model=self.MODEL,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": prompt
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_image}"
                                    }
                                }
                            ]
                        }
                    ],
                    temperature=0.3,  # Mais determinístico
                    max_tokens=200
                )
                
                content = response.choices[0].message.content.strip()
                
                # Limpar possível markdown
                if content.startswith("```json"):
                    content = content.replace("```json", "").replace("```", "")
                elif content.startswith("```"):
                    content = content.replace("```", "")
                
                data = json.loads(content)
                
                # Validar dados
                if not self._validate_data(data):
                    raise ValueError("Dados inválidos retornados pela IA")
                
                logger.info(f"Food analysis successful: {data['description']}")
                return data
                
            except RateLimitError:
                if attempt < self.MAX_RETRIES - 1:
                    st.warning(f"⏳ Rate limit atingido. Tentativa {attempt + 1}/{self.MAX_RETRIES}...")
                    continue
                else:
                    logger.error("Rate limit exceeded after retries")
                    raise
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error (attempt {attempt + 1}): {e}")
                if attempt == self.MAX_RETRIES - 1:
                    raise
                continue
            except Exception as e:
                logger.error(f"Analysis error (attempt {attempt + 1}): {e}")
                if attempt == self.MAX_RETRIES - 1:
                    raise
                continue
        
        return None
    
    @staticmethod
    def _validate_data(data: Dict) -> bool:
        """Valida dados retornados pela IA"""
        required_fields = ["description", "calories", "carbs", "protein", "fat"]
        
        if not all(field in data for field in required_fields):
            return False
        
        # Verificar se valores numéricos são válidos
        try:
            for field in ["calories", "carbs", "protein", "fat"]:
                value = float(data[field])
                if value < 0:
                    return False
        except (ValueError, TypeError):
            return False
        
        return True


# =============================
# VALIDAÇÃO DE IMAGEM
# =============================

def validate_image(image: Image.Image) -> bool:
    """Valida imagem antes de processar"""
    try:
        # Verificar dimensões
        if image.size[0] < 100 or image.size[1] < 100:
            st.error("❌ Imagem muito pequena. Use uma imagem maior que 100x100px")
            return False
        
        # Verificar tamanho
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        size_mb = len(buffer.getvalue()) / (1024 * 1024)
        
        if size_mb > 10:
            st.error("❌ Imagem muito grande. Use uma imagem menor que 10MB")
            return False
        
        return True
    except Exception as e:
        st.error(f"❌ Erro ao validar imagem: {e}")
        logger.error(f"Image validation error: {e}")
        return False


# =============================
# VISUALIZAÇÕES
# =============================

def plot_macros(data: Dict) -> plt.Figure:
    """Cria gráfico de macronutrientes"""
    macros = [data["carbs"], data["protein"], data["fat"]]
    labels = ["Carboidratos", "Proteínas", "Gorduras"]
    colors = ["#FFA500", "#4CAF50", "#FF6B6B"]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.pie(
        macros,
        labels=labels,
        autopct="%1.1f%%",
        colors=colors,
        startangle=90,
        textprops={"fontsize": 11}
    )
    ax.set_title("Distribuição de Macronutrientes", fontsize=14, fontweight="bold")
    
    return fig


def plot_summary_chart(df: pd.DataFrame) -> plt.Figure:
    """Cria gráfico de calorias por dia"""
    if df.empty:
        return None
    
    # Agrupar por dia
    df["date"] = pd.to_datetime(df["created_at"]).dt.date
    daily = df.groupby("date")["calories"].sum().sort_index()
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(daily.index.astype(str), daily.values, color="#1b8a3b", alpha=0.7)
    ax.set_xlabel("Data")
    ax.set_ylabel("Calorias")
    ax.set_title("Calorias por Dia")
    ax.grid(axis="y", alpha=0.3)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return fig


# =============================
# MAIN APP
# =============================

def main():
    # Inicializar sessão
    if "db" not in st.session_state:
        st.session_state.db = DatabaseManager()
    
    if "analyzer" not in st.session_state:
        st.session_state.analyzer = FoodAnalyzer(client)
    
    db = st.session_state.db
    analyzer = st.session_state.analyzer
    
    # Título
    st.title("🥗 Calorie Vision AI")
    st.write("Envie a foto do prato e receba estimativa nutricional em tempo real!")
    
    # Upload de imagem
    st.divider()
    st.subheader("📸 Análise de Alimentos")
    
    file = st.file_uploader(
        "Envie uma foto do prato",
        type=["jpg", "png", "jpeg"]
    )
    
    if file:
        try:
            image = Image.open(file)
            
            # Converter para RGB se necessário
            if image.mode != "RGB":
                image = image.convert("RGB")
            
            st.image(image, use_container_width=True, caption="Foto enviada")
            
            # Validar imagem
            if not validate_image(image):
                st.stop()
            
            if st.button("🔍 Analisar Alimento", use_container_width=True):
                with st.spinner("⏳ IA analisando alimento..."):
                    try:
                        result = analyzer.analyze_food(image)
                        
                        if result:
                            # Salvar no banco
                            if db.save_meal(result):
                                st.markdown("""
                                <div class='success-box'>
                                ✅ <b>Análise concluída com sucesso!</b>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Mostrar resultados
                                col1, col2 = st.columns([2, 1])
                                
                                with col1:
                                    st.markdown(f"""
                                    ### 🍽️ {result["description"]}
                                    
                                    | Nutriente | Valor |
                                    |-----------|-------|
                                    | 🔥 Calorias | {result["calories"]:.0f} kcal |
                                    | 🌾 Carboidratos | {result["carbs"]:.1f} g |
                                    | 💪 Proteínas | {result["protein"]:.1f} g |
                                    | 🧈 Gorduras | {result["fat"]:.1f} g |
                                    """)
                                
                                with col2:
                                    # Gráfico de macros
                                    fig = plot_macros(result)
                                    st.pyplot(fig, use_container_width=True)
                                
                                # Sucesso refresca a página
                                st.rerun()
                            else:
                                st.markdown("""
                                <div class='error-box'>
                                ❌ Erro ao salvar análise. Tente novamente.
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.markdown("""
                            <div class='error-box'>
                            ❌ Não foi possível analisar a imagem. Tente outra foto.
                            </div>
                            """, unsafe_allow_html=True)
                    
                    except RateLimitError:
                        st.markdown("""
                        <div class='error-box'>
                        ⚠️ Limite de requisições atingido. Aguarde alguns momentos.
                        </div>
                        """, unsafe_allow_html=True)
                    except json.JSONDecodeError:
                        st.markdown("""
                        <div class='error-box'>
                        ❌ Erro ao processar resposta da IA. Tente novamente.
                        </div>
                        """, unsafe_allow_html=True)
                    except Exception as e:
                        st.markdown(f"""
                        <div class='error-box'>
                        ❌ Erro inesperado: {str(e)[:100]}
                        </div>
                        """, unsafe_allow_html=True)
                        logger.error(f"Unexpected error: {e}", exc_info=True)
        
        except Exception as e:
            st.markdown(f"""
            <div class='error-box'>
            ❌ Erro ao abrir imagem: {str(e)}
            </div>
            """, unsafe_allow_html=True)
            logger.error(f"Image open error: {e}")
    
    # Histórico
    st.divider()
    st.header("📊 Histórico Alimentar")
    
    df = db.get_history()
    
    if len(df) > 0:
        # Resumo
        summary = db.get_summary()
        
        st.subheader("📈 Resumo Total")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("🔥 Calorias", f"{int(summary['calories'])}")
        with col2:
            st.metric("🌾 Carbs", f"{int(summary['carbs'])}g")
        with col3:
            st.metric("💪 Proteína", f"{int(summary['protein'])}g")
        with col4:
            st.metric("🧈 Gordura", f"{int(summary['fat'])}g")
        with col5:
            st.metric("🍽️ Refeições", f"{int(summary['meal_count'])}")
        
        # Gráfico de calorias por dia
        st.subheader("📉 Calorias por Dia")
        fig = plot_summary_chart(df)
        if fig:
            st.pyplot(fig, use_container_width=True)
        
        # Tabela de histórico
        st.subheader("📋 Detalhes")
        
        # Formatar para exibição
        display_df = df.copy()
        display_df["calories"] = display_df["calories"].apply(lambda x: f"{x:.0f} kcal")
        display_df["carbs"] = display_df["carbs"].apply(lambda x: f"{x:.1f}g")
        display_df["protein"] = display_df["protein"].apply(lambda x: f"{x:.1f}g")
        display_df["fat"] = display_df["fat"].apply(lambda x: f"{x:.1f}g")
        display_df["created_at"] = pd.to_datetime(display_df["created_at"]).dt.strftime("%d/%m/%Y %H:%M")
        
        # Renomear colunas
        display_df = display_df.rename(columns={
            "id": "ID",
            "description": "Prato",
            "calories": "Calorias",
            "carbs": "Carbos",
            "protein": "Proteína",
            "fat": "Gordura",
            "created_at": "Data"
        })
        
        st.dataframe(
            display_df[["ID", "Prato", "Calorias", "Carbos", "Proteína", "Gordura", "Data"]],
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("ℹ️ Nenhum prato analisado ainda. Comece enviando uma foto!")


if __name__ == "__main__":
    main()