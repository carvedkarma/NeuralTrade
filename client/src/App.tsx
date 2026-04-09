import { Switch, Route } from "wouter";
import { queryClient } from "./lib/queryClient";
import { QueryClientProvider } from "@tanstack/react-query";
import { Toaster } from "@/components/ui/toaster";
import { TooltipProvider } from "@/components/ui/tooltip";
import AppLayout from "@/components/layout/app-layout";
import CommandCenter from "@/pages/command-center";
import LiveTrading from "@/pages/live-trading";
import PaperTrading from "@/pages/paper-trading";
import Analytics from "@/pages/analytics";
import TrainingMonitor from "@/pages/training-monitor";
import TradeHistory from "@/pages/trade-history";
import SettingsPage from "@/pages/settings";
import BitgetTrading from "@/pages/bitget-trading";
import NotFound from "@/pages/not-found";
import NeuralMonitor from "@/pages/neural-monitor";
import WorldIntelligence from "@/pages/world-intelligence";

function Router() {
  return (
    <AppLayout>
      <Switch>
        <Route path="/" component={CommandCenter} />
        <Route path="/live" component={LiveTrading} />
        <Route path="/bitget" component={BitgetTrading} />
        <Route path="/paper" component={PaperTrading} />
        <Route path="/trade-history" component={TradeHistory} />
        <Route path="/analytics" component={Analytics} />
        <Route path="/training" component={TrainingMonitor} />
        <Route path="/settings" component={SettingsPage} />
        <Route path="/neural" component={NeuralMonitor} />
        <Route path="/world-intel" component={WorldIntelligence} />
        <Route component={NotFound} />
      </Switch>
    </AppLayout>
  );
}

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <TooltipProvider>
        <Toaster />
        <Router />
      </TooltipProvider>
    </QueryClientProvider>
  );
}

export default App;
