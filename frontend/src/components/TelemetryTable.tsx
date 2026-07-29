'use client';
import React, { useMemo } from 'react';

interface TelemetryPoint {
  time: number;
  altitude: number;
  speed: number;
  power_required: number;
  power_delivered: number;
  power_motor: number;
  power_engine: number;
  soc: number;
  fuel: number;
  weight: number;
  phase: string;
  deficit: number;
  u: number;
  p_aero: number;
  p_climb: number;
  climb_rate: number;
}

interface TelemetryTableProps {
  telemetry: TelemetryPoint[];
  currentIndex: number;
  onIndexChange: (index: number) => void;
}

const PHASE_COLORS: Record<string, string> = new Proxy({}, { get: () => 'text-[#5C6773]' }) as any;

const PHASE_PILL_BG: Record<string, string> = new Proxy({}, { get: () => 'bg-white/5 border-white/10 backdrop-blur-md' }) as any;

function downsampleIndices(total: number, maxRows: number): number[] {
  if (total <= maxRows) return Array.from({ length: total }, (_, i) => i);
  const step = (total - 1) / (maxRows - 1);
  const indices: number[] = [];
  for (let i = 0; i < maxRows; i++) {
    indices.push(Math.round(i * step));
  }
  return indices;
}

// Linearly interpolate a hex color between green and red by fraction (0=green,1=red)
function fuelColor(fuel: number, maxFuel: number): string {
  const frac = 1 - Math.max(0, Math.min(1, fuel / maxFuel));
  // green (#10b981) → red (#ef4444)
  const r = Math.round(16 + frac * (239 - 16));
  const g = Math.round(185 + frac * (68 - 185));
  const b = Math.round(129 + frac * (68 - 129));
  return `rgb(${r},${g},${b})`;
}

export default function TelemetryTable({ telemetry, currentIndex, onIndexChange }: TelemetryTableProps) {
  const displayIndices = useMemo(() => downsampleIndices(telemetry.length, 120), [telemetry.length]);

  const maxFuel = useMemo(() => {
    if (!telemetry.length) return 1;
    return Math.max(...telemetry.map(pt => pt.fuel));
  }, [telemetry]);

  const closestDisplayIndex = useMemo(() => {
    if (displayIndices.length === 0) return 0;
    let closest = displayIndices[0];
    let minDiff = Math.abs(closest - currentIndex);
    for (let i = 1; i < displayIndices.length; i++) {
      const diff = Math.abs(displayIndices[i] - currentIndex);
      if (diff < minDiff) { minDiff = diff; closest = displayIndices[i]; }
    }
    return closest;
  }, [displayIndices, currentIndex]);

  if (!telemetry || telemetry.length === 0) {
    return (
      <div className="w-full h-full flex items-center justify-center text-slate-500 text-xs font-mono">
        NO DATA
      </div>
    );
  }

  const formatTime = (sec: number) => {
    const h = Math.floor(sec / 3600);
    const m = Math.floor((sec % 3600) / 60);
    const s = Math.floor(sec % 60);
    return `${h.toString().padStart(2, '0')}:${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
  };

  const GRID = 'grid-cols-[60px_58px_58px_58px_52px_58px_58px_52px_62px_70px]';

  return (
    <div className="w-full h-full flex flex-col overflow-hidden">
      {/* Header */}
      <div className={`flex-shrink-0 grid ${GRID} gap-0 text-[9px] font-bold text-slate-400 uppercase tracking-wider bg-transparent px-1 py-1.5 backdrop-blur-md border-t-2 border-t-[#1F2733] border-b border-[#1F2733]`}>
        <span className="text-center">Time</span>
        <span className="text-right">P_aero</span>
        <span className="text-right">P_climb</span>
        <span className="text-right">P_req</span>
        <span className="text-center">PSR</span>
        <span className="text-right">Motor</span>
        <span className="text-right">Engine</span>
        <span className="text-right">SoC</span>
        <span className="text-right">Fuel</span>
        <span className="text-center">Phase</span>
      </div>
      {/* Unit sub-row */}
      <div className={`flex-shrink-0 grid ${GRID} gap-0 text-[8px] text-slate-600 bg-transparent px-1 py-0.5 backdrop-blur-md border-b border-[#1F2733]`}>
        <span className="text-center">–</span>
        <span className="text-right">kW</span>
        <span className="text-right">kW</span>
        <span className="text-right">kW</span>
        <span className="text-center">%</span>
        <span className="text-right">kW</span>
        <span className="text-right">kW</span>
        <span className="text-right">%</span>
        <span className="text-right">kg</span>
        <span className="text-center">–</span>
      </div>

      {/* Data rows */}
      <div className="flex-1 overflow-y-auto custom-scrollbar">
        {displayIndices.map((idx) => {
          const pt = telemetry[idx];
          if (!pt) return null;
          const isActive = idx === closestDisplayIndex;
          const socPct = pt.soc * 100;
          const fc = fuelColor(pt.fuel, maxFuel);

          return (
            <div
              key={idx}
              onClick={() => onIndexChange(idx)}
              className={`grid ${GRID} gap-0 text-[10px] font-mono px-1 py-[3px] cursor-pointer border-b border-[#1F2733] transition-colors duration-75
                odd:bg-white/[0.02]
                ${isActive
                  ? 'bg-transparent border-l-2 border-l-[#FFB454] text-[#E8EDF2]'
                  : 'text-slate-300 hover:bg-slate-800/40 border-l-2 border-l-transparent'
                }`}
            >
              <span className="text-center text-slate-400">{formatTime(pt.time)}</span>
              <span className="text-right">{pt.p_aero.toFixed(1)}</span>
              <span className={`text-right ${pt.p_climb > 0.1 ? 'text-[#FFB454]' : pt.p_climb < -0.1 ? 'text-teal-400' : ''}`}>
                {pt.p_climb.toFixed(1)}
              </span>
              <span className="text-right font-semibold">{pt.power_required.toFixed(1)}</span>
              <span className={`text-center ${pt.u > 0.4 ? 'text-[#FFB454]' : pt.u > 0.1 ? 'text-[#E8EDF2]' : 'text-[#FFB454]'}`}>
                {(pt.u * 100).toFixed(0)}%
              </span>
              <span className="text-right text-[#FFB454]">{pt.power_motor.toFixed(1)}</span>
              <span className="text-right text-[#E8EDF2]">{pt.power_engine.toFixed(1)}</span>

              {/* SoC with inline bar */}
              <span className="text-right relative">
                <span
                  className="absolute inset-0 rounded-sm opacity-20"
                  style={{
                    background: `linear-gradient(to right, ${pt.soc < 0.2 ? '#ef4444' : pt.soc > 0.7 ? '#10b981' : '#eab308'} ${socPct}%, transparent ${socPct}%)`,
                  }}
                />
                <span className={`relative z-10 ${pt.soc < 0.2 ? 'text-red-400' : pt.soc > 0.7 ? 'text-[#FFB454]' : 'text-yellow-300'}`}>
                  {socPct.toFixed(0)}%
                </span>
              </span>

              {/* Fuel: green → red interpolation */}
              <span className="text-right" style={{ color: fc }}>
                {pt.fuel.toFixed(1)}
              </span>

              {/* Phase pill */}
              <span className={`text-center capitalize px-1 py-0.5 rounded border text-[8px] font-bold ${PHASE_COLORS[pt.phase]} ${PHASE_PILL_BG[pt.phase]}`}>
                {pt.phase}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
