import PredictionForm from "./components/PredictionForm";
import FeatureImportanceChart from "./components/FeatureImportanceChart";

export default function App() {
  return (
    <div className="p-6 bg-gray-100 min-h-screen">
      <h1 className="text-3xl font-bold text-center mb-6">
        🏠 House Price Intelligence Dashboard
      </h1>
      <div className="flex gap-6 justify-center flex-wrap">
        <PredictionForm />
      </div>
    </div>
  );
}
