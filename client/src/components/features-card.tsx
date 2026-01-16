import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import type { Feature } from "@shared/schema";
import { motion } from "framer-motion";
import { Info } from "lucide-react";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";

interface FeaturesCardProps {
  features: Feature[];
}

export function FeaturesCard({ features }: FeaturesCardProps) {
  const maxImportance = Math.max(...features.map(f => Math.abs(f.importance)));

  return (
    <Card className="overflow-visible" data-testid="card-features">
      <CardHeader className="pb-3">
        <CardTitle className="text-sm font-medium text-muted-foreground" data-testid="text-features-title">
          Top Signal Drivers
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-2">
        {features.slice(0, 5).map((feature, index) => {
          const normalizedImportance = feature.importance / maxImportance;
          const isPositive = feature.importance > 0;

          return (
            <motion.div
              key={feature.name}
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: index * 0.05 }}
              className="space-y-1"
              data-testid={`feature-item-${index}`}
            >
              <div className="flex items-center justify-between gap-2">
                <div className="flex items-center gap-2 min-w-0 flex-1">
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <button className="flex items-center gap-1.5 min-w-0" data-testid={`button-feature-tooltip-${index}`}>
                        <Info className="h-3 w-3 text-muted-foreground flex-shrink-0" />
                        <span className="text-sm truncate" data-testid={`text-feature-name-${index}`}>{feature.name}</span>
                      </button>
                    </TooltipTrigger>
                    <TooltipContent side="left" className="max-w-xs">
                      <p className="text-sm">{feature.description}</p>
                    </TooltipContent>
                  </Tooltip>
                </div>
                <div className="flex items-center gap-2 flex-shrink-0">
                  <Badge 
                    variant="secondary"
                    className={`font-mono text-xs ${isPositive ? "text-emerald-400" : "text-red-400"}`}
                    data-testid={`badge-feature-value-${index}`}
                  >
                    {feature.value > 0 ? "+" : ""}{feature.value.toFixed(3)}
                  </Badge>
                </div>
              </div>
              <div className="flex h-1 gap-0.5">
                <div className="flex-1 flex justify-end">
                  {!isPositive && (
                    <motion.div
                      initial={{ width: 0 }}
                      animate={{ width: `${Math.abs(normalizedImportance) * 100}%` }}
                      transition={{ duration: 0.3 }}
                      className="h-full bg-red-500 rounded-l-full"
                      data-testid={`progress-feature-negative-${index}`}
                    />
                  )}
                </div>
                <div className="w-px bg-border" />
                <div className="flex-1">
                  {isPositive && (
                    <motion.div
                      initial={{ width: 0 }}
                      animate={{ width: `${Math.abs(normalizedImportance) * 100}%` }}
                      transition={{ duration: 0.3 }}
                      className="h-full bg-emerald-500 rounded-r-full"
                      data-testid={`progress-feature-positive-${index}`}
                    />
                  )}
                </div>
              </div>
            </motion.div>
          );
        })}
      </CardContent>
    </Card>
  );
}
