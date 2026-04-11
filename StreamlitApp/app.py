import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# Configure Streamlit page
st.set_page_config(page_title="Slang & Hate Speech Detection", layout="wide")

# Backend API URL
API_URL = "http://localhost:8000"

st.title("🛡️ Slang & Hate Speech Detection System")
st.markdown("Classify text and explain predictions using LIME and Transformer Attention Visualizations.")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Input Text")
    user_input = st.text_area("Enter text to analyze", height=150, placeholder="Type something here...")
    analyze_btn = st.button("Analyze", type="primary")

if analyze_btn and user_input:
    with st.spinner("Analyzing text..."):
        try:
            # 1. Get Predictions
            req_data = {"text": user_input}
            response = requests.post(f"{API_URL}/predict", json=req_data)
            
            if response.status_code == 200:
                predictions = response.json().get("predictions", [])
                
                # Get the highest scoring prediction id first to keep logic intact
                top_pred = max(predictions, key=lambda x: x['score'])
                top_label_idx = int(top_pred['id'].replace('LABEL_', ''))

                with col2:
                    st.subheader("Classification Results")
                    
                    # Add Neutral representation for visual charting
                    plot_preds = list(predictions)
                    neutral_score = max(0.0, 1.0 - top_pred['score'])
                    plot_preds.append({"label": "Neutral / Clean", "id": "LABEL_NEUTRAL", "score": neutral_score})
                    
                    # Prepare data for plotting
                    df_preds = pd.DataFrame(plot_preds)
                    # Sort primarily by score
                    df_preds = df_preds.sort_values(by="score", ascending=True)
                    
                    fig = px.bar(df_preds, x="score", y="label", orientation='h', 
                                 title="Confidence Scores")
                    
                    # Assign custom colors: Green for neutral, red for the rest
                    colors = ['#2ecc71' if val == 'LABEL_NEUTRAL' else '#e74c3c' for val in df_preds['id']]
                    fig.update_traces(marker_color=colors)
                    
                    fig.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0))
                    st.plotly_chart(fig, use_container_width=True)

                st.divider()
                
                # Setup Threshold
                THRESHOLD = 0.5
                if top_pred['score'] < THRESHOLD:
                    st.success(f"🟢 **Status: Clean / Neutral** (No slang/hate detected. Highest confidence was {top_pred['score']:.3f} for {top_pred['label']})")
                else:
                    st.error(f"🔴 **Status: Detected '{top_pred['label']}'** with {top_pred['score']:.3f} confidence")

                st.header("Explainability Insights")
                
                tab1, tab2 = st.tabs(["🧩 LIME Token Attributions", "🧠 Transformer Attention Visualization"])
                
                with tab1:
                    if top_pred['score'] < THRESHOLD:
                        st.markdown(f"**LIME Explanations for closest class: {top_pred['label']}** (Even though it was not detected, here is what shifted the score towards it)")
                    else:
                        st.markdown(f"**LIME Explanations for predicting class: {top_pred['label']}**")
                    with st.spinner("Generating LIME attributions..."):
                        lime_res = requests.post(
                            f"{API_URL}/explain_lime", 
                            json={"text": user_input, "target_class_idx": top_label_idx}
                        )
                        if lime_res.status_code == 200:
                            attributions = lime_res.json().get("attributions", [])
                            
                            if attributions:
                                attr_df = pd.DataFrame(attributions, columns=["Token", "Weight"])
                                attr_df = attr_df.sort_values(by="Weight", ascending=True)
                                
                                fig_lime = px.bar(
                                    attr_df, x="Weight", y="Token", orientation='h',
                                    color="Weight", color_continuous_scale="RdBu_r"
                                )
                                fig_lime.update_layout(title="Feature Importance (LIME)", height=400)
                                st.plotly_chart(fig_lime, use_container_width=True)
                            else:
                                st.info("No strong token attributions found.")
                        else:
                            st.error(f"Error getting LIME explanations: {lime_res.text}")
                            
                with tab2:
                    st.markdown("**Last Layer Average Attention Heatmap**")
                    with st.spinner("Generating Attention visualization..."):
                        attn_res = requests.post(f"{API_URL}/explain_attention", json={"text": user_input})
                        
                        if attn_res.status_code == 200:
                            attn_data = attn_res.json()
                            tokens = attn_data["tokens"]
                            attention_matrix = attn_data["attention_matrix"]
                            
                            # Create a heatmap using plotly
                            fig_attn = go.Figure(data=go.Heatmap(
                                z=attention_matrix,
                                x=tokens,
                                y=tokens,
                                colorscale='Viridis'
                            ))
                            fig_attn.update_layout(
                                title="Token-to-Token Attention (Average across Heads)",
                                height=600,
                                margin=dict(l=50, r=50, t=50, b=50),
                                xaxis_nticks=len(tokens),
                                yaxis_nticks=len(tokens)
                            )
                            # Flip y-axis for better readability
                            fig_attn.update_yaxes(autorange="reversed")
                            st.plotly_chart(fig_attn, use_container_width=True)
                        else:
                            st.error(f"Error getting Attention explanations: {attn_res.text}")

            else:
                st.error(f"Backend API error: {response.status_code}")
                
        except requests.exceptions.ConnectionError:
            st.error("Failed to connect to the backend server. Please make sure the FastAPI server is running.")
