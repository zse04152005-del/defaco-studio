import { lazy, Suspense } from "react";
import { BrowserRouter, Route, Routes } from "react-router-dom";
import { Shell } from "./components/layout/Shell";
import { InfinityLoader } from "./components/ui";

const Overview = lazy(() => import("./pages/Overview"));
const Stock = lazy(() => import("./pages/Stock"));
const Screener = lazy(() => import("./pages/Screener"));
const Factors = lazy(() => import("./pages/Factors"));
const Portfolio = lazy(() => import("./pages/Portfolio"));
const Risk = lazy(() => import("./pages/Risk"));
const Rebalance = lazy(() => import("./pages/Rebalance"));
const Backtest = lazy(() => import("./pages/Backtest"));
const Alerts = lazy(() => import("./pages/Alerts"));
const Lab = lazy(() => import("./pages/Lab"));

export default function App() {
  return (
    <BrowserRouter>
      <Shell>
        <Suspense fallback={<div className="grid h-96 place-items-center"><InfinityLoader /></div>}>
          <Routes>
            <Route path="/" element={<Overview />} />
            <Route path="/stock" element={<Stock />} />
            <Route path="/screener" element={<Screener />} />
            <Route path="/factors" element={<Factors />} />
            <Route path="/portfolio" element={<Portfolio />} />
            <Route path="/risk" element={<Risk />} />
            <Route path="/rebalance" element={<Rebalance />} />
            <Route path="/backtest" element={<Backtest />} />
            <Route path="/alerts" element={<Alerts />} />
            <Route path="/lab" element={<Lab />} />
          </Routes>
        </Suspense>
      </Shell>
    </BrowserRouter>
  );
}
