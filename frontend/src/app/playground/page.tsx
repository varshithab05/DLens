"use client";
import React, { useState } from "react";
import { Label } from "@/components/ui/label";
// import { cn } from "@/lib/utils";
import NeuralNetworkSVG from "@/components/NeuralNetworkSVG";

export default function PlaygroundForm() {
  const [file, setFile] = useState<File | null>(null);
  const [model, setModel] = useState("resnet");
  const [dataset, setDataset] = useState("imagenet");
  const [method, setMethod] = useState("grad-cam");
  const [imageResult, setImageResult] = useState<string | null>(null);
  const [prediction, setPrediction] = useState<number | null>(null);
  const [classLabel, setClassLabel] = useState<string | null>(null);
  const [confidencelvl, setConfidencelvl] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
    }
  };

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (!file) {
      alert("Please upload an image.");
      return;
    }

    setLoading(true);
    const formData = new FormData();
    formData.append("file", file);
    formData.append("model_name", `${model}_${dataset}`);
    formData.append("method", method);
    

    try {
      console.log("Form Data:", formData.getAll("model_name"));
      const response = await fetch("http://127.0.0.1:8000/run-explainability", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();
      if (data.overlay_base64) {
        setImageResult(`data:image/png;base64,${data.overlay_base64}`);
        setPrediction(data.prediction);
        setClassLabel(data.class_label);
        setConfidencelvl(data.confidence);
      }
    } catch (error) {
      console.error("Error:", error);
    }
    setLoading(false);
  };

  const handleReset = () => {
    window.location.reload();
  };

  return (
    <div className="h-screen w-full dark:bg-black bg-white dark:bg-dot-white/[0.4] bg-dot-black/[0.2] text-white flex flex-row items-center justify-center">
      <div className="max-w-md w-full mx-10 mt-5 p-4 md:p-8 bg-white dark:bg-black shadow-lg rounded-lg">
        <form onSubmit={handleSubmit} className="space-y-4">
          <LabelInputContainer>
            <Label htmlFor="model">Select Model</Label>
            <select
              id="model"
              className="border rounded-md bg-gray-900 p-2 text-white"
              value={model}
              onChange={(e) => setModel(e.target.value)}
            >
              <option value="resnet">ResNet</option>
              <option value="vgg">VGG16</option>
              <option value="mobilenet">MobileNet</option>
            </select>
          </LabelInputContainer>

          <LabelInputContainer>
          <Label htmlFor="dataset">Select Dataset</Label>
          <select
            id="dataset"
            className="border rounded-md bg-gray-900 p-2 text-white"
            value={dataset}
            onChange={(e) => setDataset(e.target.value)}
          >
            {model === "mobilenet" ? (
              <option value="mnist">MNIST</option>
            ) : <></>}
          
            <option value="imagenet">ImageNet</option>
            
          </select>
        </LabelInputContainer>

          <LabelInputContainer>
            <Label htmlFor="method">Select Explainability Method</Label>
            <select
              id="method"
              className="border rounded-md bg-gray-900 p-2 text-white"
              value={method}
              onChange={(e) => setMethod(e.target.value)}
            >
              <option value="grad-cam">Grad-CAM</option>
              <option value="lime">LIME</option>
              <option value="saliency-map">Saliency Map</option>
            </select>
          </LabelInputContainer>

          <LabelInputContainer>
            <Label htmlFor="file">Upload Image</Label>
            <input
              id="file"
              type="file"
              className="border rounded-md bg-gray-900 p-2 text-white"
              onChange={handleFileChange}
            />
          </LabelInputContainer>

          <button
            type="submit"
            className="bg-gradient-to-br from-black dark:from-zinc-900 dark:to-zinc-900 to-neutral-600 block dark:bg-zinc-800 w-full text-white rounded-md h-10 font-medium shadow-lg"
            disabled={loading}
          >
            {loading ? "Processing..." : "Predict & Explain →"}
          </button>
        </form>
      </div>

      {imageResult ? (
        <div className="mt-6 bg-gray-900 text-white p-4 rounded-xl w-1/3 shadow-lg flex flex-col items-center">
          <h3 className="text-lg font-semibold">Results:</h3>
          <img src={imageResult} alt="Explainability Result" className="mt-4 rounded-lg" />

          <div className="mt-4 text-center">
            <p className="text-lg font-semibold">Prediction: <span className="font-bold">{prediction}</span></p>
            <p className="text-lg">Class Label: <span className="font-bold">{classLabel}</span></p>
            <p className="text-lg">
              Confidence: <span className="font-bold">{(confidencelvl * 100).toFixed(2)}%</span>
            </p>
          </div>

          <button
            onClick={handleReset}
            className="mt-4 bg-red-500 hover:bg-red-600 text-white font-medium py-2 px-4 rounded-md"
          >
            Reset
          </button>
        </div>
      ) : (
        <NeuralNetworkSVG />
      )}
    </div>
  );
}

const LabelInputContainer = ({ children }: { children: React.ReactNode }) => {
  return <div className="flex flex-col space-y-2 w-full">{children}</div>;
};
