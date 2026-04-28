import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

st.set_page_config(page_title="Multilingual Abuse Detection", layout="wide")

API_URL = "https://artyuishere-multilingual-abuse-api.hf.space"

st.title("Multilingual Abuse Detection System")
st.markdown("Classify text and explain predictions using LIME and Transformer Attention Visualizations.")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Input Text")
    with st.form("analysis_form"):
        user_input = st.text_area("Enter text to analyze", height=150, placeholder="Type something here...", max_chars=2000)
        analyze_btn = st.form_submit_button("Analyze", type="primary")

if analyze_btn and user_input:
    with st.spinner("Analyzing text..."):
        try:
            pipe_res = requests.post(f"{API_URL}/pipeline", json={"text": user_input}, timeout=90)

            if pipe_res.status_code == 200:
                data = pipe_res.json()

                off_prob = data["offensive_prob"]
                is_toxic = data["is_toxic"]
                language = data["language"]
                lang_probs = data["language_probs"]

                with col2:
                    st.subheader("Classification Results")

                    plot_preds = [
                        {"label": "Toxic / Offensive", "score": off_prob},
                        {"label": "Clean / Neutral", "score": 1.0 - off_prob},
                    ]
                    df_preds = pd.DataFrame(plot_preds).sort_values("score", ascending=True)

                    fig = px.bar(df_preds, x="score", y="label", orientation='h',
                                 title="Confidence Scores")
                    colors = ['#2ecc71' if l == 'Clean / Neutral' else '#e74c3c' for l in df_preds['label']]
                    fig.update_traces(marker_color=colors)
                    fig.update_layout(height=200, margin=dict(l=0, r=0, t=30, b=0),
                                      xaxis_range=[0, 1])
                    st.plotly_chart(fig, use_container_width=True)

                    # Language bar
                    lang_df = pd.DataFrame([
                        {"Language": lang.capitalize(), "Confidence": conf}
                        for lang, conf in lang_probs.items()
                    ]).sort_values("Confidence", ascending=True)
                    fig_lang = px.bar(lang_df, x="Confidence", y="Language", orientation='h',
                                      title="Detected Language")
                    fig_lang.update_traces(marker_color='#3498db')
                    fig_lang.update_layout(height=max(180, len(lang_df) * 30), margin=dict(l=0, r=0, t=30, b=0),
                                           xaxis_range=[0, 1])
                    st.plotly_chart(fig_lang, use_container_width=True)

                st.divider()

                if is_toxic:
                    st.error(f"**Status: TOXIC / OFFENSIVE** - "
                             f"Confidence: {off_prob:.1%} | Language: {language.capitalize()}")
                else:
                    st.success(f"**Status: CLEAN** - "
                               f"Confidence: {1 - off_prob:.1%} | Language: {language.capitalize()}")

     
                with st.expander("Explore Processing Pipeline Details"):
                    st.markdown("See exactly how your text flows through each stage of the model.")

                    st.markdown("---")
                    st.markdown("### Raw Input")
                    st.code(data["raw_text"], language=None)

                    st.markdown("### Preprocessing")
                    st.markdown("*URLs removed, emojis converted, repeated chars collapsed, abbreviations expanded, lowercased*")

                    pcol1, pcol2 = st.columns(2)
                    with pcol1:
                        st.markdown("**Before:**")
                        st.code(data["raw_text"], language=None)
                    with pcol2:
                        st.markdown("**After:**")
                        st.code(data["preprocessed_text"], language=None)

                    st.markdown("### Tokenization (XLM-RoBERTa SentencePiece)")
                    st.markdown(f"Text split into **{data['num_raw_tokens']}** subword tokens by the SentencePiece tokenizer.")

                    token_html = ""
                    for i, (tok, tid) in enumerate(zip(data["raw_tokens"], data["raw_token_ids"])):
                        bg = "#2c3e50" if i % 2 == 0 else "#34495e"
                        if tok in ("<s>", "</s>"):
                            bg = "#8e44ad"  
                        token_html += (
                            f'<span style="display:inline-block; background:{bg}; color:#ecf0f1; '
                            f'padding:4px 8px; margin:2px; border-radius:6px; font-family:monospace; '
                            f'font-size:13px;" title="ID: {tid}">{tok}</span>'
                        )
                    st.markdown(token_html, unsafe_allow_html=True)
                    st.caption("Hover over tokens to see their numeric IDs. Purple = special tokens <s> and </s>.")

                    st.markdown("### Padding & Attention Mask")
                    st.markdown(
                        f"Sequence padded to **{data['max_length']}** tokens: "
                        f"**{data['num_real_tokens']}** real + "
                        f"**{data['num_padded_tokens']}** padding."
                    )

                    fig_pad = go.Figure()
                    fig_pad.add_trace(go.Bar(
                        x=[data["num_real_tokens"]], y=["Sequence"],
                        orientation='h', name="Real Tokens",
                        marker_color="#2ecc71", text=[f"{data['num_real_tokens']} real"],
                        textposition="inside",
                    ))
                    fig_pad.add_trace(go.Bar(
                        x=[data["num_padded_tokens"]], y=["Sequence"],
                        orientation='h', name="Padding [PAD]",
                        marker_color="#7f8c8d", text=[f"{data['num_padded_tokens']} pad"],
                        textposition="inside",
                    ))
                    fig_pad.update_layout(
                        barmode='stack', height=100,
                        margin=dict(l=0, r=0, t=0, b=0),
                        showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=1),
                        xaxis_title=f"Total: {data['max_length']} tokens",
                    )
                    st.plotly_chart(fig_pad, use_container_width=True)

                    mask_sample = data["attention_mask_sample"]
                    mask_html = '<div style="font-family:monospace; font-size:12px; line-height:1.8;">'
                    mask_html += '<b>Attention Mask (first 20):</b> '
                    for m in mask_sample:
                        bg = "#2ecc71" if m == 1 else "#e74c3c"
                        mask_html += f'<span style="background:{bg}; color:white; padding:2px 6px; margin:1px; border-radius:3px;">{m}</span>'
                    mask_html += ' ...'
                    mask_html += '</div>'
                    st.markdown(mask_html, unsafe_allow_html=True)
                    st.caption("1 = real token (model reads it), 0 = padding (model ignores it)")

                    st.markdown("### Model Inference")
                    st.markdown("The padded input flows through the multi-task transformer:")

                    import textwrap
                    arch_html = textwrap.dedent("""
                    <div style="text-align:center; padding:15px;">
                        <div style="display:inline-block; text-align:center;">
                            <!-- Input -->
                            <div style="background:#2c3e50; color:#ecf0f1; padding:10px 30px; border-radius:8px; display:inline-block; font-weight:600;">
                                input_ids + attention_mask
                            </div>
                            <div style="color:#95a5a6; font-size:22px;">|</div>
                            <div style="color:#95a5a6; font-size:22px;">V</div>

                            <!-- Backbone -->
                            <div style="background:linear-gradient(135deg, #2980b9, #3498db); color:white; padding:15px 40px; border-radius:10px; display:inline-block; font-weight:600; font-size:16px; box-shadow: 0 4px 15px rgba(52,152,219,0.3);">
                                XLM-RoBERTa-Large<br>
                                <span style="font-size:12px; opacity:0.8;">24 layers x 16 heads x 1024 hidden</span>
                            </div>
                            <div style="color:#95a5a6; font-size:22px;">|</div>
                            <div style="color:#95a5a6; font-size:22px;">V</div>

                            <!-- CLS -->
                            <div style="background:#8e44ad; color:white; padding:8px 25px; border-radius:8px; display:inline-block; font-weight:600;">
                                [CLS] token -> Dropout(0.1) -> 1024-dim vector
                            </div>

                            <div style="display:flex; justify-content:center; gap:40px; margin-top:5px;">
                                <div style="text-align:center;">
                                    <div style="color:#95a5a6; font-size:22px;">|</div>
                                    <div style="background:#e74c3c; color:white; padding:10px 20px; border-radius:8px; font-weight:600;">
                                        Offensive Head<br>
                                        <span style="font-size:12px;">Linear(1024 -> 1) -> Sigmoid</span>
                                    </div>
                                </div>
                                <div style="text-align:center;">
                                    <div style="color:#95a5a6; font-size:22px;">|</div>
                                    <div style="background:#3498db; color:white; padding:10px 20px; border-radius:8px; font-weight:600;">
                                        Language Head<br>
                                        <span style="font-size:12px;">Linear(1024 -> {len(data['language_probs'])}) -> Softmax</span>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                    """)
                    st.markdown(arch_html.replace('\n', ''), unsafe_allow_html=True)

                    st.markdown("### Raw Model Outputs")

                    out1, out2 = st.columns(2)
                    with out1:
                        st.markdown("**Offensive Head**")
                        st.markdown(f"- Raw logit: `{data['offensive_logit']}`")
                        st.markdown(f"- After sigmoid: **`{data['offensive_prob']}`**")
                        st.markdown(f"- Threshold: `0.5`")
                        st.markdown(f"- Verdict: **{'TOXIC' if is_toxic else 'CLEAN'}**")

                    with out2:
                        st.markdown("**Language Head**")
                        lang_keys = list(lang_probs.keys())
                        for i, lang_name in enumerate(lang_keys):
                            logit_val = data["language_logits"][i]
                            prob_val = lang_probs[lang_name]
                            marker = "> " if lang_name.lower() == language else "  "
                            st.markdown(f"{marker}**{lang_name.capitalize()}**: logit=`{logit_val}` -> prob=**`{prob_val}`**")

                    if data.get("attention_words"):
                        st.markdown("### What the Model Focused On")
                        st.markdown("CLS-token attention from the last transformer layer, averaged across all 16 heads.")

                        words = data["attention_words"]
                        weights = data["attention_weights"]

                        hl_html = '<div style="background:#1a1a2e; border-radius:10px; padding:15px; line-height:2.2; font-size:15px; margin:8px 0;">'
                        for w, wt in zip(words, weights):
                            r = int(231 * wt + 40 * (1 - wt))
                            g = int(76 * wt + 40 * (1 - wt)) if is_toxic else int(204 * wt + 40 * (1 - wt))
                            b = int(60 * wt + 50 * (1 - wt)) if is_toxic else int(113 * wt + 50 * (1 - wt))
                            alpha = 0.2 + wt * 0.7
                            color = "#fff" if wt > 0.3 else "#aaa"
                            hl_html += (
                                f'<span style="display:inline-block; background:rgba({r},{g},{b},{alpha:.2f}); '
                                f'color:{color}; padding:3px 7px; margin:2px; border-radius:5px; '
                                f'font-family:sans-serif;" title="attention: {wt:.3f}">{w}</span>'
                            )
                        hl_html += '</div>'
                        st.markdown(hl_html, unsafe_allow_html=True)
                        st.caption("Brighter = higher attention. Hover for exact values.")

                        attn_df = pd.DataFrame({"Word": words, "Attention": weights}).sort_values("Attention", ascending=True)
                        color_scale = "Reds" if is_toxic else "Greens"
                        fig_attn = px.bar(attn_df, x="Attention", y="Word", orientation='h',
                                          color="Attention", color_continuous_scale=color_scale)
                        fig_attn.update_layout(
                            title="Word Attention Scores",
                            height=max(250, len(words) * 25),
                            margin=dict(l=0, r=0, t=30, b=0),
                            coloraxis_showscale=False, xaxis_range=[0, 1.05],
                        )
                        st.plotly_chart(fig_attn, use_container_width=True)


                st.header("Explainability Insights")
                tab1, tab2 = st.tabs(["LIME Token Attributions", "Transformer Attention Visualization"])

                with tab1:
                    st.markdown(f"**LIME Explanations for offensive class**")
                    with st.spinner("Generating LIME attributions..."):
                        lime_res = requests.post(
                            f"{API_URL}/explain_lime",
                            json={"text": user_input, "target_class_idx": 1},
                        )
                        if lime_res.status_code == 200:
                            attributions = lime_res.json().get("attributions", [])
                            if attributions:
                                attr_df = pd.DataFrame(attributions, columns=["Token", "Weight"])
                                attr_df = attr_df.sort_values(by="Weight", ascending=True)
                                fig_lime = px.bar(attr_df, x="Weight", y="Token", orientation='h',
                                                   color="Weight", color_continuous_scale="RdBu_r")
                                fig_lime.update_layout(
                                    title="Feature Importance (LIME)", 
                                    height=400,
                                    paper_bgcolor="rgba(0,0,0,0)",
                                    plot_bgcolor="rgba(0,0,0,0)",
                                    font=dict(color="#f8fafc")
                                )
                                st.plotly_chart(fig_lime, use_container_width=True)
                            else:
                                st.info("No strong token attributions found.")
                        else:
                            st.error(f"Error getting LIME explanations: {lime_res.text}")

                with tab2:
                    st.markdown("**Last Layer Average Attention Heatmap**")
                    with st.spinner("Generating Attention visualization..."):
                        attn_res = requests.post(f"{API_URL}/explain_attention", json={"text": user_input}, timeout=60)
                        if attn_res.status_code == 200:
                            attn_data = attn_res.json()
                            tokens = attn_data["tokens"]
                            attention_matrix = attn_data["attention_matrix"]

                            n = len(tokens)
                            cell_size = 90  
                            fig_size = max(500, n * cell_size)

                            fig_hm = go.Figure(data=go.Heatmap(
                                z=attention_matrix, x=tokens, y=tokens,
                                colorscale='Viridis',
                                xgap=1, ygap=1,
                            ))
                            fig_hm.update_layout(
                                title="Token-to-Token Attention (Average across Heads)",
                                height=fig_size,
                                margin=dict(l=100, r=20, t=60, b=100),
                                xaxis=dict(
                                    tickangle=-45,
                                    tickfont=dict(size=12, color="#f8fafc"),
                                    title="",
                                ),
                                yaxis=dict(
                                    tickfont=dict(size=12, color="#f8fafc"),
                                    title="",
                                    autorange="reversed",
                                ),
                                paper_bgcolor="rgba(0,0,0,0)",
                                plot_bgcolor="rgba(0,0,0,0)",
                                font=dict(color="#f8fafc"),
                            )
                            st.plotly_chart(fig_hm, use_container_width=True)
                        else:
                            st.error(f"Error getting Attention explanations: {attn_res.text}")

            else:
                st.error(f"Backend API error: {pipe_res.status_code} - {pipe_res.text}")

        except requests.exceptions.ConnectionError:
            st.error("Cannot reach the cloud server. Check your internet connection or the HF Space status.")
        except requests.exceptions.Timeout:
            st.warning("The Hugging Face server was sleeping and is now waking up (this takes ~60 seconds). Please click Analyze again in a moment!")
