'use client';

import React, { useState, useEffect, useMemo, useCallback, useRef } from 'react';
import dynamic_import from 'next/dynamic';
import TelemetryTable from './TelemetryTable';
import TelemetryChart from './TelemetryChart';

const FlightScene = dynamic_import(() => import('./FlightScene'), { ssr: false });

interface OptimalSpecs {
  engine_kw: number;
  battery_kwh: number;
  motor_kw: number;
  endurance_hours: number;
  endurance_hours_min?: number;
  endurance_hours_max?: number;
  empty_weight_kg: number;
  fuel_weight_kg: number;
  total_weight_kg: number;
  motor_model: string;
  engine_weight_kg: number;
  motor_weight_kg: number;
  battery_weight_kg: number;
}

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
  sfc: number;
}

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:8000';

function fmtTime(sec: number): string {
  const h = Math.floor(sec / 3600);
  const m = Math.floor((sec % 3600) / 60);
  const s = Math.floor(sec % 60);
  if (h > 0) return `${h}h ${m}m`;
  return `${m}m ${s}s`;
}

const PHASE_BADGE: Record<string, string> = {
  takeoff:   'bg-red-900/60 text-red-300 border-red-700/40',
  climb:     'bg-amber-900/60 text-[#FFB454] border-amber-700/40',
  cruise:    'bg-cyan-900/60 text-[#E8EDF2] border-cyan-700/40',
  loiter:    'bg-violet-900/60 text-violet-300 border-violet-700/40',
  descent:   'bg-teal-900/60 text-teal-300 border-teal-700/40',
  landing:   'bg-emerald-900/60 text-[#FFB454] border-[#1F2733]',
  completed: 'bg-slate-800/60 text-[#5C6773] border-[#1F2733]',
};

const PHASE_GLOW: Record<string, string> = new Proxy({}, { get: () => '' }) as any;

// HAL logo SVG (hexagon outline)
function HalLogo() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#FFB454" strokeWidth="1.5" strokeLinejoin="round">
      <polygon points="12 2 21.39 7 21.39 17 12 22 2.61 17 2.61 7" />
      <polygon points="12 6 17.5 9 17.5 15 12 18 6.5 15 6.5 9" opacity="0.4"/>
    </svg>
  );
}

export default function Dashboard() {
  const [targetSpeedKmh, setTargetSpeedKmh] = useState<number>(250);
  const [targetAltitude, setTargetAltitude] = useState<number>(5000);
  const [payloadWeight, setPayloadWeight] = useState<number>(200);
  const [enableLoiter, setEnableLoiter] = useState<boolean>(true);
  const [showMatrix, setShowMatrix] = useState<boolean>(true);
  const [initialFuelFraction, setInitialFuelFraction] = useState<number>(1.0);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [specs, setSpecs] = useState<OptimalSpecs | null>(null);
  const [telemetry, setTelemetry] = useState<TelemetryPoint[]>([]);
  const [currentIndex, setCurrentIndex] = useState<number>(0);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [activeTab, setActiveTab] = useState<'3d' | 'charts'>('3d');
  // Cosmetic generation counter for loading screen
  const [genCount, setGenCount] = useState<number>(1);
  const [loadProgress, setLoadProgress] = useState<number>(0);
  const genTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const currentPoint = useMemo(() => {
    if (telemetry.length === 0) return null;
    return telemetry[Math.min(currentIndex, telemetry.length - 1)];
  }, [telemetry, currentIndex]);

  useEffect(() => {
    if (!isPlaying || telemetry.length === 0) return;
    const interval = setInterval(() => {
      setCurrentIndex(prev => {
        if (prev >= telemetry.length - 1) { setIsPlaying(false); return prev; }
        return prev + 1;
      });
    }, 30);
    return () => clearInterval(interval);
  }, [isPlaying, telemetry.length]);

  // Drive cosmetic generation counter while loading
  useEffect(() => {
    if (loading) {
      setGenCount(1);
      setLoadProgress(0);
      genTimerRef.current = setInterval(() => {
        setGenCount(prev => (prev < 15 ? prev + 1 : 15));
        setLoadProgress(prev => Math.min(prev + 6.5, 98));
      }, 500);
    } else {
      if (genTimerRef.current) clearInterval(genTimerRef.current);
      setLoadProgress(100);
    }
    return () => {
      if (genTimerRef.current) clearInterval(genTimerRef.current);
    };
  }, [loading]);

  const handleOptimize = useCallback(async () => {
    setLoading(true);
    setError(null);
    setCurrentIndex(0);
    setIsPlaying(false);
    try {
      const response = await fetch(`${API_URL}/api/optimize`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          target_speed_kmh: targetSpeedKmh,
          target_altitude: targetAltitude,
          payload_weight: payloadWeight,
          enable_loiter: enableLoiter,
          initial_fuel_fraction: initialFuelFraction,
        }),
      });
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Optimization failed.');
      }
      const data = await response.json();
      setSpecs(data.optimal_specs);
      setTelemetry(data.telemetry);
    } catch (err: any) {
      console.error(err);
      setError(err.message || 'Backend connection failed.');
    } finally {
      setLoading(false);
    }
  }, [targetSpeedKmh, targetAltitude, payloadWeight, enableLoiter, initialFuelFraction]);

  useEffect(() => { handleOptimize(); }, []);

  const phaseDurations = useMemo(() => {
    if (!telemetry.length) return null;
    const phases: Record<string, { start: number; end: number }> = {};
    for (const pt of telemetry) {
      if (!phases[pt.phase]) phases[pt.phase] = { start: pt.time, end: pt.time };
      phases[pt.phase].end = pt.time;
    }
    return phases;
  }, [telemetry]);

  const totalMissionTime = useMemo(() => {
    if (!phaseDurations) return 1;
    return Object.values(phaseDurations).reduce((acc, t) => acc + (t.end - t.start), 0) || 1;
  }, [phaseDurations]);

  const weightBreakdown = useMemo(() => {
    if (!specs) return null;
    return [
      { name: 'Airframe', weight: 350, color: '#4B5563' },
      { name: 'Payload', weight: payloadWeight, color: '#6366F1' },
      { name: 'Turboshaft', weight: specs.engine_weight_kg, color: '#3B82F6' },
      { name: 'EMRAX Motor', weight: specs.motor_weight_kg, color: '#10B981' },
      { name: 'Battery', weight: specs.battery_weight_kg, color: '#F59E0B' },
      { name: 'Fuel (Jet-A1)', weight: specs.fuel_weight_kg, color: '#EF4444' },
    ];
  }, [specs, payloadWeight]);

  const totalWeight = useMemo(() => weightBreakdown?.reduce((s, b) => s + b.weight, 0) || 1000, [weightBreakdown]);

  return (
    <div className="h-screen w-screen text-[#E8EDF2] flex flex-col relative overflow-hidden select-none">

      {/* ── TOP BAR (glassmorphism) ─────────────────────────────────────── */}
      <header
        className="h-11 flex-shrink-0 flex items-center justify-between px-4 border-b border-[#1F2733]"
        style={{
          background: 'rgba(10,14,20,0.65)',
          backdropFilter: 'blur(24px)',
          WebkitBackdropFilter: 'blur(12px)',
          boxShadow: 'none',
        }}
      >
        <div className="flex items-center gap-3">
          <HalLogo />
          <span className="text-[11px] font-extrabold tracking-[0.22em] uppercase text-[#FFB454]">AeroOptima</span>
          <span className="hidden sm:block text-[9px] text-[#5C6773] font-mono border-l border-slate-700 pl-3 ml-1">HAL × IIT Indore | PS-1 Hybrid-Electric UAV</span>
        </div>
        <div className="flex items-center gap-4 text-[10px] font-mono text-[#5C6773]">
          {specs && (
            <>
              <span>ENDURANCE: <b className="text-[#FFB454] font-mono">
                {specs.endurance_hours.toFixed(2)}h
                {specs.endurance_hours_min !== undefined && specs.endurance_hours_max !== undefined && (
                  <span className="text-[#5C6773] text-[9px] font-normal ml-1">
                    ({specs.endurance_hours_min.toFixed(1)}-{specs.endurance_hours_max.toFixed(1)}h range)
                  </span>
                )}
              </b></span>
              <span>ENGINE: <b className="text-[#E8EDF2]">{specs.engine_kw.toFixed(1)}kW</b></span>
              <span>BATT: <b className="text-[#FFB454]">{specs.battery_kwh.toFixed(1)}kWh</b></span>
              <span>MOTOR: <b className="text-[#E8EDF2]">{specs.motor_model}</b></span>
              {/* SYS ONLINE badge */}
              <span className="flex items-center gap-1.5 ml-2 border border-[#1F2733] rounded-full px-2 py-0.5 bg-[#0A0E14]">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 " />
                <span className="text-[9px] font-bold text-[#FFB454] tracking-widest">SYS ONLINE</span>
              </span>
            </>
          )}
        </div>
      </header>

      {/* ── MAIN 3-PANEL GRID ──────────────────────────────────────────── */}
      <div className="flex-1 flex overflow-hidden">

        {/* ── PANEL A: Controls (Left Sidebar) ──────────────────────────── */}
        <aside className="w-[284px] flex-shrink-0 border-r border-[#1F2733] bg-[#0A0E14]/40 backdrop-blur-xl flex flex-col overflow-y-auto custom-scrollbar">

          {/* ── Simulation Constraints ─── */}
          <div className="bg-white/5 border border-white/10 rounded-lg backdrop-blur-md shadow-[0_4px_24px_rgba(0,0,0,0.2)] m-3 p-5 panel-enter" style={{ animationDelay: '0ms' }}>
            <h2 className="text-[9px] font-bold text-[#5C6773] uppercase tracking-[0.15em] mb-3 flex items-center gap-1.5">
              <span>⚡</span> Simulation Constraints
            </h2>

            <div className="mb-3">
              <div className="flex justify-between text-[10px] mb-1">
                <span className="text-[#5C6773]">Cruise Speed</span>
                <span className="font-mono text-[#FFB454] font-bold bg-white/10 px-1.5 rounded backdrop-blur-md shadow-inner">{targetSpeedKmh} km/h</span>
              </div>
              <input type="range" min="150" max="350" step="5" value={targetSpeedKmh}
                onChange={(e) => setTargetSpeedKmh(parseInt(e.target.value))} disabled={loading}
                className="w-full h-1 bg-slate-800 rounded appearance-none cursor-pointer accent-[#FFB454] disabled:opacity-40" />
            </div>

            <div className="mb-3">
              <div className="flex justify-between text-[10px] mb-1">
                <span className="text-[#5C6773]">Target Altitude</span>
                <span className="font-mono text-[#FFB454] font-bold bg-white/10 px-1.5 rounded backdrop-blur-md shadow-inner">{targetAltitude}m</span>
              </div>
              <input type="range" min="3000" max="10000" step="250" value={targetAltitude}
                onChange={(e) => setTargetAltitude(parseInt(e.target.value))} disabled={loading}
                className="w-full h-1 bg-slate-800 rounded appearance-none cursor-pointer accent-[#FFB454] disabled:opacity-40" />
            </div>

            <div className="mb-3">
              <div className="flex justify-between text-[10px] mb-1">
                <span className="text-[#5C6773]">Payload Mass</span>
                <span className="font-mono text-[#FFB454] font-bold bg-white/10 px-1.5 rounded backdrop-blur-md shadow-inner">{payloadWeight} kg</span>
              </div>
              <input type="range" min="100" max="300" step="5" value={payloadWeight}
                onChange={(e) => setPayloadWeight(parseInt(e.target.value))} disabled={loading}
                className="w-full h-1 bg-slate-800 rounded appearance-none cursor-pointer accent-[#FFB454] disabled:opacity-40" />
            </div>

            <div className="mb-3">
              <div className="flex justify-between text-[10px] mb-1">
                <span className="text-[#5C6773]">Initial Fuel Load</span>
                <span className="font-mono text-orange-400 font-bold bg-white/10 px-1.5 rounded backdrop-blur-md shadow-inner">{Math.round(initialFuelFraction * 100)}%</span>
              </div>
              <input type="range" min="0.1" max="1.0" step="0.05" value={initialFuelFraction}
                onChange={(e) => setInitialFuelFraction(parseFloat(e.target.value))} disabled={loading}
                className="w-full h-1 bg-slate-800 rounded appearance-none cursor-pointer accent-orange-500 disabled:opacity-40" />
            </div>

            <div className="flex items-center gap-2 mb-4">
              <input type="checkbox" id="loiterToggle" checked={enableLoiter}
                onChange={(e) => setEnableLoiter(e.target.checked)} disabled={loading}
                className="w-3.5 h-3.5 rounded accent-[#FFB454] cursor-pointer disabled:opacity-40" />
              <label htmlFor="loiterToggle" className="text-[10px] text-[#5C6773] cursor-pointer">Enable Loiter Phase (Orbit)</label>
            </div>

            {/* Execute button with shimmer */}
            <div className="relative">
              <button
                onClick={handleOptimize}
                disabled={loading}
                className="w-full py-2.5 text-[11px] font-bold uppercase tracking-[0.18em] rounded border transition-all overflow-hidden relative
                  bg-[#12161F] border-[#1F2733] text-[#FFB454]
                  hover:bg-[#1F2733] hover:border-[#1F2733]
                  hover:shadow-none
                  disabled:opacity-40 disabled:pointer-events-none
                  flex items-center justify-center gap-2 group"
              >
                {/* Shimmer sweep on hover */}
                <span
                  className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-300 pointer-events-none"
                  style={{
                    background: 'linear-gradient(90deg, transparent 0%, rgba(255,255,255,0.12) 50%, transparent 100%)',
                    backgroundSize: '200% 100%',
                    animation: 'shimmer 1.4s linear infinite',
                  }}
                />
                {loading
                  ? (<><div className="w-3.5 h-3.5 border-2 border-emerald-400 border-t-transparent rounded-full animate-spin" />OPTIMIZING...</>)
                  : <>▶ EXECUTE GA OPTIMIZER</>
                }
              </button>
              {/* Progress bar under button */}
              {loading && (
                <div className="mt-1 h-0.5 w-full bg-slate-800 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-[#FFB454] rounded-full transition-all duration-500"
                    style={{ width: `${loadProgress}%` }}
                  />
                </div>
              )}
            </div>
          </div>

          {/* ── Current State ─── */}
          {currentPoint && !loading && (
            <div className="bg-white/5 border border-white/10 rounded-lg backdrop-blur-md shadow-[0_4px_24px_rgba(0,0,0,0.2)] m-3 p-5 panel-enter" style={{ animationDelay: '50ms' }}>
              <h2 className="text-[9px] font-bold text-[#5C6773] uppercase tracking-[0.15em] mb-2 flex items-center gap-1.5">
                <span>📡</span> Current State — T+{fmtTime(currentPoint.time)}
              </h2>
              <div className={`inline-block text-[9px] font-bold uppercase px-2 py-0.5 rounded border mb-3 ${PHASE_BADGE[currentPoint.phase] || PHASE_BADGE.completed} ${PHASE_GLOW[currentPoint.phase] || ''}`}>
                {currentPoint.phase}
              </div>
              <div className="grid grid-cols-2 gap-1.5 text-[10px]">
                {[
                  { label: 'ALT', value: `${currentPoint.altitude.toFixed(0)}m`, cls: '' },
                  { label: 'TAS', value: `${(currentPoint.speed * 3.6).toFixed(0)} km/h`, cls: '' },
                  { label: 'P_REQ', value: `${currentPoint.power_required.toFixed(1)} kW`, cls: '' },
                  { label: 'ELEC%', value: `${(currentPoint.u * 100).toFixed(0)}%`, cls: currentPoint.u > 0.4 ? 'text-amber-400' : currentPoint.u > 0.05 ? 'text-cyan-400' : 'text-[#5C6773]' },
                  { label: 'MOTOR', value: `${currentPoint.power_motor.toFixed(1)} kW`, cls: 'text-[#FFB454]' },
                  { label: 'ENGINE', value: `${currentPoint.power_engine.toFixed(1)} kW`, cls: 'text-[#E8EDF2]' },
                  { label: 'SOC', value: `${(currentPoint.soc * 100).toFixed(1)}%`, cls: currentPoint.soc < 0.2 ? 'text-red-400' : 'text-yellow-300' },
                  { label: 'FUEL', value: `${currentPoint.fuel.toFixed(1)} kg`, cls: currentPoint.fuel < 30 ? 'text-red-400' : 'text-slate-100' },
                ].map(({ label, value, cls }) => (
                  <div key={label} className="bg-[#12161F] rounded-md p-1.5 border border-[#1F2733]">
                    <span className="text-[#5C6773] text-[8px] block leading-none mb-0.5">{label}</span>
                    <p className={`font-mono font-bold text-[10px] leading-none ${cls}`}>{value}</p>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* ── Mission Profile Timeline ─── */}
          {phaseDurations && !loading && (
            <div className="bg-white/5 border border-white/10 rounded-lg backdrop-blur-md shadow-[0_4px_24px_rgba(0,0,0,0.2)] m-3 p-5 panel-enter" style={{ animationDelay: '100ms' }}>
              <h2 className="text-[9px] font-bold text-[#5C6773] uppercase tracking-[0.15em] mb-3 flex items-center gap-1.5">
                <span>🗺️</span> Mission Profile
              </h2>
              <div className="relative pl-4">
                {/* Vertical timeline line */}
                <div className="absolute left-1.5 top-1.5 bottom-1.5 w-px bg-gradient-to-b from-emerald-500/60 via-slate-700/40 to-transparent" />
                {Object.entries(phaseDurations)
                  .filter(([p]) => p !== 'completed')
                  .map(([phase, t], idx) => {
                    const durSec = t.end - t.start;
                    const durMin = durSec / 60;
                    const fraction = Math.min(durSec / totalMissionTime, 1);
                    return (
                      <div key={phase} className="mb-2.5 last:mb-0">
                        {/* Dot + label row */}
                        <div className="flex items-center justify-between mb-1" style={{ animationDelay: `${idx * 40}ms` }}>
                          <div className="flex items-center gap-2">
                            <span className="absolute left-0.5 w-2 h-2 rounded-full bg-emerald-500 border border-emerald-300 shadow-[0_0_6px_rgba(255,180,84,0.7)]"
                              style={{ marginTop: 0 }} />
                            <span className="capitalize font-medium text-[10px] text-slate-300">{phase}</span>
                          </div>
                          <span className="text-[#5C6773] font-mono text-[10px]">{durMin.toFixed(1)}m</span>
                        </div>
                        {/* Mini progress bar */}
                        <div className="h-1 bg-slate-800/80 rounded-full overflow-hidden">
                          <div
                            className="h-full rounded-full bg-[#FFB454] transition-all"
                            style={{ width: `${fraction * 100}%` }}
                          />
                        </div>
                      </div>
                    );
                  })}
              </div>
            </div>
          )}

          {/* ── Weight Budget ─── */}
          {weightBreakdown && !loading && (
            <div className="bg-white/5 border border-white/10 rounded-lg backdrop-blur-md shadow-[0_4px_24px_rgba(0,0,0,0.2)] m-3 p-5 panel-enter" style={{ animationDelay: '150ms' }}>
              <h2 className="text-[9px] font-bold text-[#5C6773] uppercase tracking-[0.15em] mb-2 flex items-center gap-1.5">
                <span>⚖️</span> MTOW Budget — {specs?.total_weight_kg} kg
              </h2>
              <div className="h-4 w-full flex rounded-full overflow-hidden mb-2.5 ">
                {weightBreakdown.map((bar, i) => (
                  <div
                    key={i}
                    style={{ width: `${(bar.weight / totalWeight) * 100}%`, backgroundColor: bar.color }}
                    title={`${bar.name}: ${bar.weight.toFixed(1)} kg`}
                  />
                ))}
              </div>
              {weightBreakdown.map((bar, i) => (
                <div key={i} className="flex justify-between text-[10px] py-0.5">
                  <span className="flex items-center gap-1.5">
                    <span className="w-2 h-2 rounded-sm flex-shrink-0" style={{ backgroundColor: bar.color }} />
                    <span className="text-[#5C6773]">{bar.name}</span>
                  </span>
                  <span className="font-mono text-[#E8EDF2]">{bar.weight.toFixed(1)} kg</span>
                </div>
              ))}
            </div>
          )}

          {/* ── System Constants ─── */}
          {specs && !loading && (
            <div className="bg-white/5 border border-white/10 rounded-lg backdrop-blur-md shadow-[0_4px_24px_rgba(0,0,0,0.2)] m-3 p-5 panel-enter" style={{ animationDelay: '200ms' }}>
              <h2 className="text-[9px] font-bold text-[#5C6773] uppercase tracking-[0.15em] mb-2 flex items-center gap-1.5">
                <span>🔬</span> System Constants
              </h2>
              <div className="space-y-1 text-[10px]">
                <div className="flex justify-between"><span className="text-[#5C6773]">Drag Polar</span><span className="font-mono text-slate-300">Oswald AR=16.07</span></div>
                <div className="flex justify-between"><span className="text-[#5C6773]">SFC</span><span className="font-mono text-slate-300">{currentPoint && currentPoint.sfc !== undefined ? `${currentPoint.sfc.toFixed(3)} kg/kWh` : '0.380 kg/kWh'}</span></div>
                <div className="flex justify-between"><span className="text-[#5C6773]">C-Rate Limit</span><span className="font-mono text-yellow-400">3C/5C</span></div>
                <div className="flex justify-between"><span className="text-[#5C6773]">Motor η</span><span className="font-mono text-slate-300">96%</span></div>
                <div className="flex justify-between"><span className="text-[#5C6773]">Strategy</span><span className="font-mono text-[#FFB454]">Zhang et al.</span></div>
              </div>
            </div>
          )}

          {error && (
            <div className="m-2">
              <div className="bg-red-950/40 border border-red-800/40 text-red-300 rounded-lg p-2.5 text-[10px]">
                <span className="font-bold">ERR:</span> {error}
              </div>
            </div>
          )}
        </aside>

        {/* ── PANEL B + C ──────────────────────────────────────────────── */}
        <div className="flex-1 flex flex-col overflow-hidden">

          {/* Tab Bar */}
          <div className="h-9 flex-shrink-0 flex items-end border-b border-[#1F2733] bg-transparent px-2 gap-1">
            {/* Animated underline tabs */}
            {(['3d', 'charts'] as const).map((tab) => (
              <button
                key={tab}
                onClick={() => setActiveTab(tab)}
                className={`px-3 pb-1.5 pt-1 text-[10px] font-bold uppercase tracking-wider transition-all relative border-b-2 ${
                  activeTab === tab
                    ? 'border-b-[#FFB454] text-[#FFB454]'
                    : 'border-b-transparent text-[#5C6773] hover:text-slate-300'
                }`}
              >
                {tab === '3d' ? '3D Flight Profile' : 'Telemetry Charts'}
              </button>
            ))}
            <button
              onClick={() => setShowMatrix(!showMatrix)}
              className="px-2 py-0.5 text-[9px] font-bold uppercase tracking-wider rounded border border-slate-800 text-[#5C6773] hover:text-[#E8EDF2] hover:bg-slate-800/40 transition-colors ml-2 mb-1.5"
            >
              {showMatrix ? 'Collapse Matrix' : 'Expand Matrix'}
            </button>
            <div className="flex-1" />
            {currentPoint && (
              <span className="text-[10px] font-mono text-[#5C6773] mb-1.5">
                T+{fmtTime(currentPoint.time)} | {currentPoint.altitude.toFixed(0)}m ALT | {currentPoint.phase.toUpperCase()}
              </span>
            )}
          </div>

          {/* Content */}
          <div className="flex-1 flex overflow-hidden">
            <div className="flex-1 flex flex-col overflow-hidden">
              {loading ? (
                /* ── Tactical loading screen ──────────────────────────── */
                <div className="flex-1 flex flex-col items-center justify-center bg-[#0A0E14] gap-4">
                  {/* Spinning ring */}
                  <div className="relative w-20 h-20">
                    <div className="absolute inset-0 rounded-full border-4 border-slate-800" />
                    <div className="absolute inset-0 rounded-full border-4 border-t-emerald-500 border-r-transparent border-b-transparent border-l-transparent animate-spin" />
                    <div className="absolute inset-2 rounded-full border-2 border-t-transparent border-r-cyan-500/50 border-b-transparent border-l-transparent animate-spin" style={{ animationDuration: '1.5s', animationDirection: 'reverse' }} />
                  </div>
                  <div className="text-center">
                    <p className="font-extrabold text-base text-slate-100 tracking-[0.3em] uppercase">Genetic Algorithm Executing</p>
                    <p className="text-[11px] font-mono text-[#FFB454] mt-1 tracking-widest">
                      GEN {String(genCount).padStart(2, '0')} / 15
                    </p>
                    <p className="text-[10px] text-[#5C6773] mt-1 font-mono">Sizing propulsion architecture | 40 pop × 15 gen</p>
                  </div>
                  <div className="w-64 h-1.5 bg-slate-800 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-[#FFB454] rounded-full transition-all duration-500"
                      style={{ width: `${loadProgress}%` }}
                    />
                  </div>
                </div>
              ) : activeTab === '3d' ? (
                <>
                  <div className="flex-1 relative">
                    <FlightScene telemetry={telemetry} currentIndex={currentIndex} />
                    {/* Legend glassmorphism pill */}
                    <div
                      className="absolute bottom-3 left-3 flex gap-3 text-[9px] font-mono px-3 py-1.5 rounded-full border border-slate-700/50"
                      style={{
                        background: 'rgba(11,15,25,0.72)',
                        backdropFilter: 'blur(10px)',
                        WebkitBackdropFilter: 'blur(10px)',
                        boxShadow: '0 2px 16px rgba(0,0,0,0.5)',
                      }}
                    >
                      <span className="flex items-center gap-1.5"><span className="w-2 h-2 rounded-full bg-amber-500" />BATTERY</span>
                      <span className="flex items-center gap-1.5"><span className="w-2 h-2 rounded-full bg-cyan-500" />HYBRID</span>
                      <span className="flex items-center gap-1.5"><span className="w-2 h-2 rounded-full bg-emerald-500" />ENGINE</span>
                      <span className="flex items-center gap-1.5"><span className="w-2 h-2 rounded-full bg-slate-500" />IDLE</span>
                    </div>
                  </div>

                  {/* ── Scrubber bar ─── */}
                  <div className="h-12 flex-shrink-0 flex items-center gap-3 px-4 border-t border-[#1F2733] bg-[#12161F]">
                    <button
                      onClick={() => setIsPlaying(!isPlaying)}
                      className="w-8 h-8 flex items-center justify-center text-sm border border-slate-700 rounded-full text-slate-300
                        hover:bg-[#1F2733] hover:shadow-none transition-all"
                    >
                      {isPlaying ? '⏸' : '▶'}
                    </button>
                    <div className="flex-1 relative">
                      <input
                        type="range"
                        min="0"
                        max={Math.max(0, telemetry.length - 1)}
                        step="1"
                        value={currentIndex}
                        onChange={(e) => { setCurrentIndex(parseInt(e.target.value)); setIsPlaying(false); }}
                        className="w-full appearance-none cursor-pointer"
                        style={{
                          height: '4px',
                          background: telemetry.length > 0
                            ? `linear-gradient(to right, #10b981 ${(currentIndex / Math.max(1, telemetry.length - 1)) * 100}%, #22d3ee ${(currentIndex / Math.max(1, telemetry.length - 1)) * 100}%, #1e293b 100%)`
                            : '#1e293b',
                          borderRadius: '9999px',
                        }}
                      />
                    </div>
                    <span className="text-[10px] font-mono text-[#5C6773] w-20 text-right">
                      {telemetry.length > 0 ? `${currentIndex + 1}/${telemetry.length}` : '0/0'}
                    </span>
                  </div>
                </>
              ) : (
                <div className="flex-1 overflow-y-auto p-2 custom-scrollbar">
                  <TelemetryChart telemetry={telemetry} />
                </div>
              )}
            </div>

            {showMatrix && (
              <div className="w-[620px] flex-shrink-0 border-l border-[#1F2733] bg-[#0A0E14]/40 backdrop-blur-xl flex flex-col overflow-hidden">
                <div className="h-7 flex-shrink-0 flex items-center px-2 border-b border-[#1F2733]">
                  <h3 className="text-[9px] font-bold text-[#5C6773] uppercase tracking-[0.15em]">Propulsion Status Matrix — kW</h3>
                </div>
                <TelemetryTable
                  telemetry={telemetry}
                  currentIndex={currentIndex}
                  onIndexChange={(i) => { setCurrentIndex(i); setIsPlaying(false); }}
                />
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
