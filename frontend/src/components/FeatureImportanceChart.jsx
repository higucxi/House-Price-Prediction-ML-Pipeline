import Highcharts from "highcharts";
import HighchartsReact from "highcharts-react-official";
import { useEffect, useState } from "react";
import axios from "axios";

export default function FeatureImportanceChart() {
  const [data, setData] = useState([]);

  useEffect(() => {
    axios.get("http://127.0.0.1:8000/feature-importance")
        .then(res => {
        setData(res.data.map((d) => [d.feature, d.importance]));
        })
        .catch(err => console.error(err));
  }, []);


  const options = {
    chart: { type: "bar" },
    title: { text: "Top 10 Important Features" },
    xAxis: { categories: data.map((d) => d.feature) },
    yAxis: { title: { text: "Importance" } },
    series: [{ data: data.map((d) => d.importance), name: "Feature Importance" }],
  };

  return (
    <div className="bg-white p-4 rounded-xl shadow-md">
      <HighchartsReact highcharts={Highcharts} options={options} />
    </div>
  );
}
