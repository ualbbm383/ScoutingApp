from pathlib import Path
import streamlit as st

from src.data_downloader import ensure_data_files
from src.build_top5_events import build_top5_events
from src.update_player_metrics import update_player_metrics
from src.update_team_metrics import update_team_metrics
from src.update_player_clusters import update_player_clusters
from src.update_all import update_all
from src.enrich_player_metrics import enrich_player_metrics
from src.auth import check_login

if not check_login():
    st.stop()

# DOBLE OPCIÓN DE ESTAR EN LOCAL O EN NUBE, PARA ACCEDER A ARCHIVOS LOCALES O GOOGLE DRIVE
try:
    IS_CLOUD = bool(st.secrets["IS_CLOUD"])
except Exception:
    IS_CLOUD = False

if IS_CLOUD:
    ensure_data_files()

st.set_page_config(page_title="Scouting App", layout="wide")

# --------------------------------------------------
# ESTILO GENERAL
# --------------------------------------------------

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }

    .hero-box {
        padding: 1.8rem 1.8rem 1.4rem 1.8rem;
        border-radius: 20px;
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: white;
        margin-bottom: 1.4rem;
    }

    .hero-title {
        font-size: 2.35rem;
        font-weight: 700;
        margin-bottom: 0.35rem;
    }

    .hero-subtitle {
        font-size: 1.02rem;
        opacity: 0.93;
        line-height: 1.6;
    }

    .section-card {
        padding: 1.2rem 1.2rem 1rem 1.2rem;
        border-radius: 16px;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        height: 100%;
    }

    .section-title {
        font-size: 1.08rem;
        font-weight: 700;
        margin-bottom: 0.45rem;
        color: #0f172a;
    }

    .section-text {
        font-size: 0.97rem;
        line-height: 1.58;
        color: #334155;
    }

    .mini-note {
        font-size: 0.93rem;
        color: #475569;
        line-height: 1.55;
    }

    .footer-box {
        padding: 1rem 1.2rem;
        border-radius: 14px;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        height: 100%;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# HERO
# --------------------------------------------------
st.markdown(
    """
    <div class="hero-box">
        <div class="hero-title">⚽ Herramienta de Scouting y Análisis</div>
        <div class="hero-subtitle">
            Herramienta de análisis de jugadores y equipos basada en <b>event data</b> de las cinco grandes ligas.
            El proyecto combina métricas de rendimiento, visualización avanzada y técnicas de clustering
            para identificar <b>perfiles de jugador</b> y <b>estilos colectivos de juego</b>.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# MÓDULOS PRINCIPALES
# --------------------------------------------------
st.markdown("## Módulos principales")

col1, col2, col3, col4 = st.columns(4)

with col4:
    st.markdown(
        """
        <div class="section-card">
            <div class="section-title">👤 Informe del Jugador</div>
            <div class="section-text">
                Consulta el perfil individual de cualquier jugador de la base de datos.
                Incluye métricas agregadas, campogramas de eventos, estilo detectado
                y distribución híbrida de perfiles.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        """
        <div class="section-card">
            <div class="section-title">🧠 Estilos de Jugadores</div>
            <div class="section-text">
                Clasificación automática por posición a partir del estilo de juego.
                Permite detectar perfiles como organizadores, regateadores o delanteros referencia.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col1:
    st.markdown(
        """
        <div class="section-card">
            <div class="section-title">🛡️ Estilos de Equipos</div>
            <div class="section-text">
                Análisis colectivo de los equipos para identificar modelos de juego con balón
                y comportamiento sin balón, combinando clustering, PPDA y métricas tácticas.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        """
        <div class="section-card">
            <div class="section-title">📊 Análisis de Métricas</div>
            <div class="section-text">
                Explora rankings dinámicos, scatter plots personalizables y comparaciones radar entre jugadores,
                con filtros por liga, temporada, posición, perfil y minutos.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")

st.markdown(
        """
        <div class="section-card">
            <div class="section-title">🛠️ Corrección y Mantenimiento</div>
            <div class="section-text">
                Posibilidad de corregir manualmente posiciones de jugadores, actualizar datos al instante,
                reconstruir métricas y recalcular clusters para mantener la herramienta alineada
                con la realidad competitiva.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.divider()

# --------------------------------------------------
# FLUJO DE USO
# --------------------------------------------------
st.markdown("## Cómo utilizar la herramienta")

flow1, flow2, flow3 = st.columns(3)

with flow1:
    st.markdown(
        """
        <div class="section-card">
            <div class="section-title">1. Selecciona un módulo</div>
            <div class="section-text">
                Usa el menú lateral para acceder al bloque que quieras explorar:
                perfil individual, clustering de jugadores, estilos de equipo o análisis de métricas.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with flow2:
    st.markdown(
        """
        <div class="section-card">
            <div class="section-title">2. Ajusta los filtros</div>
            <div class="section-text">
                Filtra por posición, liga, temporada, minutos o perfil para adaptar el análisis
                al contexto competitivo que te interese.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with flow3:
    st.markdown(
        """
        <div class="section-card">
            <div class="section-title">3. Interpreta el resultado</div>
            <div class="section-text">
                Combina métricas, gráficos y clustering para construir una visión más rica
                del jugador o equipo, tanto desde el punto de vista cuantitativo como táctico.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.divider()

# --------------------------------------------------
# ACTUALIZACIÓN
# --------------------------------------------------
st.markdown("## Actualización de datos")

if not IS_CLOUD:
    st.caption("Estos procesos reconstruyen la base analítica y son el núcleo operativo de la aplicación.")

    col_up1, col_up2 = st.columns(2)

    with col_up1:
        st.markdown(
            """
            <div class="section-card">
                <div class="section-title">🔄 Actualización completa</div>
                <div class="section-text">
                    Descarga nuevos eventos desde el scraper, reconstruye el dataset top 5,
                    actualiza métricas de jugador y equipo, vuelve a enriquecer la metadata
                    y recalcula los clusters de jugadores.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button("Actualizar todo", width="stretch"):
            with st.spinner("Descargando eventos, consolidando top 5, reconstruyendo métricas y recalculando clusters..."):
                try:
                    update_all()
                    st.cache_data.clear()
                    st.success("Actualización completa terminada correctamente.")
                except Exception as e:
                    st.error(f"Error al actualizar: {e}")

    with col_up2:
        st.markdown(
            """
            <div class="section-card">
                <div class="section-title">♻️ Recalcular base analítica</div>
                <div class="section-text">
                    Reconstruye el dataset consolidado top 5 a partir de los parquet por liga,
                    actualiza métricas, vuelve a enriquecer el dataset de jugadores
                    y recalcula los clusters sin necesidad de volver a scrapear.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button("Actualizar métricas y clusters", width="stretch"):
            with st.spinner("Reconstruyendo top 5, métricas, metadata y clusters..."):
                try:
                    build_top5_events()
                    update_player_metrics()
                    update_team_metrics()
                    enrich_player_metrics()
                    update_player_clusters()
                    st.cache_data.clear()
                    st.success("Top 5, métricas, metadata y clusters actualizados correctamente.")
                except Exception as e:
                    st.error(f"Error al actualizar métricas: {e}")

else:
    st.caption("Modo nube: la aplicación funciona en modo visualización y carga los datos automáticamente desde almacenamiento externo.")
    st.info("Las actualizaciones de scraping, métricas y clustering se realizan en local y después se reflejan en la versión desplegada.")

st.divider()

# --------------------------------------------------
# SOBRE EL PROYECTO
# --------------------------------------------------
st.markdown("## Sobre el proyecto")

st.markdown(
        """
        <div class="footer-box">
            <div class="section-text">
                Esta aplicación ha sido desarrollada como <b>Trabajo de Fin de Máster en Big Data Aplicado al Scouting en Fútbol</b>,
                con el objetivo de construir una herramienta útil para direcciones deportivas que permita acceder de manera fácil, intuitiva y personalizable a los datos de cualquier jugador de las cinco grandes ligas.
                Se busca profundizar un poco en los datos clásicos y aportar una herramienta que <b>identifique perfiles de jugador</b>
                y <b>detecte los diferentes estilos de cada equipo</b>, para combinarlos con el análisis más convencional de métricas.<br><br>
                <b>Creador</b>: Santiago Sesma Núñez.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )



# Código para hacer cambios en la nube a partir de ahora

# git add .
# git commit -m "Describe el cambio"
# git push