import { BrowserRouter, Route, Routes } from "react-router-dom";
import { Footer } from "./components/layout/Footer";
import { Navbar } from "./components/layout/Navbar";
import { ClassificationPredictPage } from "./pages/ClassificationPredictPage";
import { DashboardPage } from "./pages/DashboardPage";
import { HomePage } from "./pages/HomePage";
import { PricelistPage } from "./pages/PricelistPage";
import { RegressionPredictPage } from "./pages/RegressionPredictPage";
import { SchedulingScriptPage } from "./pages/SchedulingScriptPage";
import { TeamPage } from "./pages/TeamPage";

function App() {
  return (
    <BrowserRouter>
      <div className="flex min-h-screen flex-col">
        <Navbar />
        <main className="flex-1">
          <Routes>
            <Route path="/" element={<HomePage />} />
            <Route path="/dashboard" element={<DashboardPage />} />
            <Route path="/predict/classification" element={<ClassificationPredictPage />} />
            <Route path="/predict/regression" element={<RegressionPredictPage />} />
            <Route path="/scheduling-script" element={<SchedulingScriptPage />} />
            <Route path="/pricelist" element={<PricelistPage />} />
            <Route path="/team" element={<TeamPage />} />
          </Routes>
        </main>
        <Footer />
      </div>
    </BrowserRouter>
  );
}

export default App;
