from typing import Optional, Union

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, RedirectResponse
import uvicorn
import json
import os
import pandas as pd
from live_infer import infer_with_fallback

class App:
    def __init__(self):
        self.app = FastAPI()
        self.register_routes()

    def register_routes(self):
        self.app.get("/")(self.read_root)
        self.app.get("/deployer")(self.deployer)
        self.app.get("/inference_page")(self.inference_page)

    async def __call__(self, scope, receive, send):
        return await self.app(scope, receive, send)

    async def read_root(self) -> HTMLResponse:
        with open("website/index.html", "r") as f:
            return HTMLResponse(f.read())

    async def deployer(self, dataset : str) -> HTMLResponse:
        with open("website/deployer.html") as f:
            template = f.read()

            if os.path.exists(f"website/cached_acc_cost_pairs/{dataset}.pq"):
                curve_points = pd.read_parquet(f"website/cached_acc_cost_pairs/{dataset}.pq")
                accuracies = curve_points['acc'].apply(lambda acc: f"{acc:.1%}").tolist()
                costs = curve_points['cost'].apply(lambda cost: f"{cost:.2f}").tolist()
            else:
                # TODO Alex: make this work as a loading page
                accuracies = list(range(100))
                costs = list(range(100))

            html = template\
                .replace("PLACEHOLDER_ACCURACIES", json.dumps(accuracies))\
                .replace("PLACEHOLDER_COSTS", json.dumps(costs))\
                .replace("PLACEHOLDER_DATASET", dataset)
            return HTMLResponse(html)

    async def inference_page(self, dataset : str, confidence_threshold : int, question : Optional[str] = None) -> HTMLResponse:
        assert dataset
        assert 0 <= confidence_threshold <= 100
        if question is None or not question.strip():
            question = ""

        explanation, confidence, model = infer_with_fallback(question, round(confidence_threshold / 100))

        hidden = 'hidden="true"'
        if question:
            hidden = ""

        with open("website/inference_page.html", "r") as f:
            template = f.read()
            html = template\
                .replace("PLACEHOLDER_DATASET", dataset)\
                .replace("PLACEHOLDER_QUESTION", question)\
                .replace("PLACEHOLDER_CONFIDENCE_THRESHOLD", str(confidence_threshold))\
                .replace("PLACEHOLDER_ANSWER", explanation)\
                .replace("PLACEHOLDER_CONFIDENCE", str(confidence))\
                .replace("PLACEHOLDER_HIDDEN", hidden)\
                .replace("PLACEHOLDER_MODEL", model)
            return HTMLResponse(html)


if __name__ == "__main__":
    app = App()
    uvicorn.run(app)
