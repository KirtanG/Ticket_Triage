import gradio as gr
import requests
from typing import Tuple

BASE_URL = "http://localhost:8080/api/v1"

def classify_ticket(text: str) -> Tuple[dict, str]:
    try:
        if not text or not text.strip():
            return {}, "**Error:** Ticket description cannot be empty."

        response = requests.post(
            f"{BASE_URL}/predict",
            json={"text": text, "return_all_scores": True},
            timeout=10
        )
        response.raise_for_status()
        data = response.json()
        
        scores = data["all_scores"]
        metadata = f"**Predicted:** {data['predicted_class']} ({data['confidence']:.2%})\n"
        
        return scores, metadata
        
    except requests.exceptions.ConnectionError:
        return {}, "**Error:** Cannot connect to API. Is the backend running?"
    except Exception as e:
        return {}, f"**Error:** {str(e)}"


def classify_batch(tickets_text: str) -> str:
    try:
        # Split textarea input into individual tickets (one per line), strip blanks
        tickets = [t.strip() for t in tickets_text.strip().splitlines() if t.strip()]
        
        if not tickets:
            return "**Error:** No tickets provided. Enter one ticket per line."
        
        response = requests.post(
            f"{BASE_URL}/predict/batch",
            json={"tickets": tickets},
            timeout=30
        )
        response.raise_for_status()
        data = response.json()

        predictions = data.get("predictions", [])
        
        # Build a readable markdown table of results
        lines = ["| # | Ticket | Predicted Class |",
                 "|---|--------|----------------|"]
        
        for i, (ticket, pred) in enumerate(zip(tickets, predictions), start=1):
            # Handle both dict-style and plain string predictions
            if isinstance(pred, dict):
                label = pred.get("predicted_class", "N/A")
            else:
                label    = str(pred)
            
            # Truncate long tickets for display
            display_ticket = (ticket[:60] + "…") if len(ticket) > 60 else ticket
            lines.append(f"| {i} | {display_ticket} | **{label}** |")
        
        summary = f"**{len(predictions)} ticket(s) classified successfully.**\n\n"
        return summary + "\n".join(lines)
    
    except requests.exceptions.ConnectionError:
        return "**Error:** Cannot connect to API. Is the backend running?"
    except Exception as e:
        return f"**Error:** {str(e)}"


# Single prediction examples
SINGLE_EXAMPLES = [
    "My laptop won't turn on and the power light is not working",
    "Need access to the finance shared drive for Q4 reports",
    "Request to purchase 2 new monitors for the design team",
    "HR portal login not working, forgot my password",
    r"Server storage is at 95% capacity, need more disk space",
]

# Batch prediction examples (5 tickets pre-filled, one per line)
BATCH_EXAMPLES = [
    "My laptop won't turn on and the power light is not working",
    "Need access to the finance shared drive for Q4 reports",
    "Request to purchase 2 new monitors for the design team",
    "HR portal login not working, forgot my password",
    r"Server storage is at 95% capacity, need more disk space",
]
BATCH_EXAMPLE_TEXT = "\n".join(BATCH_EXAMPLES)


with gr.Blocks(title="IT Ticket Classifier") as demo:
    gr.Markdown("# IT Support Ticket Classification")
    gr.Markdown("Powered by fine-tuned **ModernBERT** with FP16 quantization")

    with gr.Tabs():

        # ── Tab 1: Single Prediction ──────────────────────────────────────────
        with gr.Tab("Online Prediction"):
            with gr.Row():
                with gr.Column():
                    text_input = gr.Textbox(
                        lines=5,
                        placeholder="Enter IT support ticket description...",
                        label="Ticket Description"
                    )
                    submit_btn = gr.Button("Classify Ticket", variant="primary")

                with gr.Column():
                    label_output = gr.Label(num_top_classes=8, label="Classification Scores")
                    metadata_output = gr.Markdown()

            gr.Examples(examples=SINGLE_EXAMPLES, inputs=text_input)

            submit_btn.click(
                fn=classify_ticket,
                inputs=text_input,
                outputs=[label_output, metadata_output]
            )

        # ── Tab 2: Batch Prediction ───────────────────────────────────────────
        with gr.Tab("Batch Prediction"):
            gr.Markdown(
                "Enter one ticket per line. "
                "Click **Load Examples** to pre-fill with 5 test tickets, "
                "or type your own."
            )
            with gr.Row():
                with gr.Column():
                    batch_input = gr.Textbox(
                        lines=10,
                        placeholder="Ticket one\nTicket two\nTicket three\n...",
                        label="Tickets (one per line)"
                    )
                    with gr.Row():
                        load_examples_btn = gr.Button("Load Examples")
                        batch_submit_btn  = gr.Button("Run Batch", variant="primary")

                with gr.Column():
                    batch_output = gr.Markdown(label="Batch Results")

            load_examples_btn.click(
                fn=lambda: BATCH_EXAMPLE_TEXT,
                inputs=None,
                outputs=batch_input
            )

            batch_submit_btn.click(
                fn=classify_batch,
                inputs=batch_input,
                outputs=batch_output
            )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860,theme=gr.themes.Glass())