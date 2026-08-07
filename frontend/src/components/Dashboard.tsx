'use client';

import React, { useState, useEffect, useMemo, useCallback, useRef } from 'react';
import dynamic_import from 'next/dynamic';
import TelemetryTable from './TelemetryTable';
import TelemetryChart from './TelemetryChart';

const FlightScene = dynamic_import(() => import('./FlightScene'), { ssr: false });

interface LegResult {
  role: string;
  required: number;
  achieved: number;
  completed: boolean;
}

interface OptimalSpecs {
  engine_kw: number;
  battery_kwh: number;
  motor_kw: number;
  motor_count: number;
  endurance_hours: number;
  endurance_hours_min?: number;
  endurance_hours_max?: number;
  on_station_min_required: number;
  on_station_min_achieved: number;
  bonus_reserve_loiter_min: number;
  leg_results: LegResult[];
  reserve_fuel_equivalent_min: number;
  reserve_battery_equivalent_min: number;
  engine_out_survivable: boolean;
  empty_weight_kg: number;
  empty_weight_fraction: number;
  fuel_weight_kg: number;
  total_weight_kg: number;
  motor_model: string;
  engine_weight_kg: number;
  motor_weight_kg: number;
  battery_weight_kg: number;
  battery_chemistry: string;
  generator_architecture: string;
  generator_efficiency: number;
  l_over_d_max: number;
  power_split_policy?: { cruise: number; loiter: number } | null;
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

type LegRole = 'cruise' | 'loiter';

interface MissionLeg {
  id: string;
  role: LegRole;
  altitude_m: number;
  speed_kmh: number;
  distance_km: number;   // used when role === 'cruise'
  duration_min: number;  // used when role === 'loiter'
  windKmh: number;       // signed: positive=headwind, negative=tailwind (cruise legs only)
}

const DEFAULT_LEGS: MissionLeg[] = [
  { id: 'leg-1', role: 'cruise', altitude_m: 5000, speed_kmh: 250, distance_km: 300, duration_min: 60, windKmh: 0 },
  { id: 'leg-2', role: 'loiter', altitude_m: 3000, speed_kmh: 180, distance_km: 300, duration_min: 60, windKmh: 0 },
];

const API_URL = process.env.NEXT_PUBLIC_API_URL ||
  (typeof window !== 'undefined'
    ? `http://${window.location.hostname}:8000`
    : 'http://localhost:8000');

function fmtTime(sec: number): string {
  const h = Math.floor(sec / 3600);
  const m = Math.floor((sec % 3600) / 60);
  const s = Math.floor(sec % 60);
  if (h > 0) return `${h}h ${m}m`;
  return `${m}m ${s}s`;
}

// Rough client-side stall-speed estimate for the leg-speed advisory — deliberately
// approximate (assumes a representative 1000kg weight; actual weight depends on
// whatever engine/battery/motor-count the GA ultimately picks). Not a hard validation,
// just a heads-up before a full GA run is spent on an input that would stall-terminate.
function approxStallKmh(altitudeM: number): number {
  const rho0 = 1.225;
  const ratio = Math.max(0.05, 1 - 2.25577e-5 * altitudeM);
  const rho = rho0 * Math.pow(ratio, 4.25588);
  const S = 14.0, CLmax = 1.5, g = 9.81, weightKg = 1000;
  const vStallMs = Math.sqrt((2 * weightKg * g) / (rho * S * CLmax));
  return vStallMs * 3.6;
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

const PHASE_GLOW = new Proxy({} as Record<string, string>, {
  get: () => ''
});

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
  const [legs, setLegs] = useState<MissionLeg[]>(DEFAULT_LEGS);
  const legIdCounter = useRef<number>(DEFAULT_LEGS.length);

  const [baseElevationM, setBaseElevationM] = useState<number>(0);
  const [ambientTempC, setAmbientTempC] = useState<number>(15);
  const [turbulenceLevel, setTurbulenceLevel] = useState<number>(0);
  const [silentLoiterMode, setSilentLoiterMode] = useState<boolean>(true);
  const [batteryChemistry, setBatteryChemistry] = useState<string>('Li-NCA');
  const [optimizePowerSplit, setOptimizePowerSplit] = useState<boolean>(false);
  const [showAdvanced, setShowAdvanced] = useState<boolean>(false);

  const [payloadWeight, setPayloadWeight] = useState<number>(200);
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
  const [loadProgress, setLoadProgress] = useState<number>(100);
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
      genTimerRef.current = setInterval(() => {
        setGenCount(prev => (prev < 15 ? prev + 1 : 15));
        setLoadProgress(prev => Math.min(prev + 6.5, 98));
      }, 500);
    } else {
      if (genTimerRef.current) clearInterval(genTimerRef.current);
    }
    return () => {
      if (genTimerRef.current) clearInterval(genTimerRef.current);
    };
  }, [loading]);

  const addLeg = useCallback((role: LegRole) => {
    legIdCounter.current += 1;
    setLegs(prev => {
      const lastAltitude = prev.length ? prev[prev.length - 1].altitude_m : 5000;
      return [...prev, {
        id: `leg-${legIdCounter.current}`,
        role,
        altitude_m: lastAltitude,
        speed_kmh: role === 'cruise' ? 250 : 180,
        distance_km: 200,
        duration_min: 30,
        windKmh: 0,
      }];
    });
  }, []);

  const removeLeg = useCallback((id: string) => {
    setLegs(prev => (prev.length > 1 ? prev.filter(l => l.id !== id) : prev));
  }, []);

  const updateLeg = useCallback((id: string, patch: Partial<MissionLeg>) => {
    setLegs(prev => prev.map(l => (l.id === id ? { ...l, ...patch } : l)));
  }, []);

  const legWarnings = useMemo(() => {
    return legs
      .map((leg, idx) => {
        const minSafeKmh = 1.2 * approxStallKmh(leg.altitude_m);
        if (leg.speed_kmh < minSafeKmh) {
          return `Leg ${idx + 1} (${leg.role}): ${leg.speed_kmh}km/h is below the ~${minSafeKmh.toFixed(0)}km/h stall-safety margin at ${leg.altitude_m}m (rough estimate).`;
        }
        return null;
      })
      .filter((w): w is string => w !== null);
  }, [legs]);

  const handleOptimize = useCallback(async () => {
    setLoading(true);
    setError(null);
    setCurrentIndex(0);
    setIsPlaying(false);
    setGenCount(1);
    setLoadProgress(0);
    try {
      const apiLegs = legs.map(leg => {
        const payload: Record<string, unknown> = {
          role: leg.role,
          altitude_m: leg.altitude_m,
          speed_kmh: leg.speed_kmh,
        };
        if (leg.role === 'cruise') {
          payload.distance_km = leg.distance_km;
          payload.headwind_kmh = leg.windKmh;
        } else {
          payload.duration_min = leg.duration_min;
        }
        return payload;
      });

      const response = await fetch(`${API_URL}/api/optimize`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          legs: apiLegs,
          base_elevation_m: baseElevationM,
          payload_weight: payloadWeight,
          initial_fuel_fraction: initialFuelFraction,
          ambient_temp_c: ambientTempC,
          turbulence_level: turbulenceLevel,
          silent_loiter_mode: silentLoiterMode,
          battery_chemistry: batteryChemistry,
          optimize_power_split: optimizePowerSplit,
        }),
      });
      if (!response.ok) {
        const errorData = await response.json();
        const detail = Array.isArray(errorData.detail)
          ? errorData.detail.map((d: { msg?: string }) => d.msg).join('; ')
          : errorData.detail;
        throw new Error(detail || 'Optimization failed.');
      }
      const data = await response.json();
      setSpecs(data.optimal_specs);
      setTelemetry(data.telemetry);
    } catch (err: unknown) {
      console.error(err);
      const errMsg = err instanceof Error ? err.message : 'Backend connection failed.';
      setError(errMsg);
    } finally {
      setLoading(false);
      setLoadProgress(100);
    }
  }, [legs, baseElevationM, payloadWeight, initialFuelFraction, ambientTempC, turbulenceLevel, silentLoiterMode, batteryChemistry, optimizePowerSplit]);

  useEffect(() => {
    const timer = setTimeout(() => {
      handleOptimize();
    }, 0);
    return () => clearTimeout(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Contiguous same-phase segments (NOT grouped by phase name globally) — a multi-leg
  // mission can revisit "cruise"/"loiter"/"climb"/"descent" several times, so grouping by
  // name alone would collapse distinct ingress/egress/extend segments into one misleading
  // start->end span.
  const phaseSegments = useMemo(() => {
    if (!telemetry.length) return null;
    const segments: { phase: string; start: number; end: number }[] = [];
    for (const pt of telemetry) {
      const last = segments[segments.length - 1];
      if (last && last.phase === pt.phase) {
        last.end = pt.time;
      } else {
        segments.push({ phase: pt.phase, start: pt.time, end: pt.time });
      }
    }
    return segments;
  }, [telemetry]);

  const totalMissionTime = useMemo(() => {
    if (!phaseSegments) return 1;
    return phaseSegments.reduce((acc, s) => acc + (s.end - s.start), 0) || 1;
  }, [phaseSegments]);

  const weightBreakdown = useMemo(() => {
    if (!specs) return null;
    return [
      { name: 'Airframe', weight: 350, color: '#4B5563' },
      { name: 'Payload', weight: payloadWeight, color: '#6366F1' },
      { name: 'Turboshaft', weight: specs.engine_weight_kg, color: '#3B82F6' },
      { name: `EMRAX Motor ×${specs.motor_count}`, weight: specs.motor_weight_kg, color: '#10B981' },
      { name: 'Battery', weight: specs.battery_weight_kg, color: '#F59E0B' },
      { name: 'Fuel (Jet-A1)', weight: specs.fuel_weight_kg, color: '#EF4444' },
    ];
  }, [specs, payloadWeight]);

  const totalWeight = useMemo(() => weightBreakdown?.reduce((s, b) => s + b.weight, 0) || 1000, [weightBreakdown]);

  const onStationMet = useMemo(() => {
    if (!specs) return true;
    return specs.on_station_min_required === 0 || specs.on_station_min_achieved >= specs.on_station_min_required;
  }, [specs]);

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
              <span>ON-STATION: <b className={onStationMet ? 'text-emerald-400' : 'text-red-400'}>
                {specs.on_station_min_achieved.toFixed(0)}/{specs.on_station_min_required.toFixed(0)}min {onStationMet ? '✓' : '✗'}
              </b></span>
              <span>ENGINE: <b className="text-[#E8EDF2]">{specs.engine_kw.toFixed(1)}kW</b></span>
              <span>BATT: <b className="text-[#FFB454]">{specs.battery_kwh.toFixed(1)}kWh</b></span>
              <span>MOTOR: <b className="text-[#E8EDF2]">{specs.motor_count}× {specs.motor_model}</b></span>
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
        <aside className="w-[284px] flex-shrink-0 border-r border-[#1F2733] bg-[#0A0E14] flex flex-col overflow-y-auto custom-scrollbar">

          {/* ── Simulation Constraints ─── */}
          <div className="bg-[#12161F] border border-[#1F2733] rounded-lg shadow-md m-3 p-5 panel-enter" style={{ animationDelay: '0ms' }}>
            <h2 className="text-[9px] font-bold text-[#5C6773] uppercase tracking-[0.15em] mb-3 flex items-center gap-1.5">
              <span>⚡</span> Simulation Constraints
            </h2>

            {legs.map((leg, idx) => (
              <div key={leg.id} className={idx > 0 ? 'mt-3 pt-3 border-t border-[#1F2733]' : ''}>
                <div className="flex items-center justify-between mb-2">
                  <div className="relative">
                    <select
                      value={leg.role}
                      disabled={loading}
                      onChange={(e) => updateLeg(leg.id, { role: e.target.value as LegRole })}
                      className="appearance-none bg-[#0A0E14] border border-[#1F2733] rounded text-[10px] font-bold uppercase tracking-wide text-[#E8EDF2] pl-2 pr-5 py-1 cursor-pointer disabled:opacity-40 hover:border-[#FFB454]/40 focus:outline-none focus:border-[#FFB454]/60"
                    >
                      <option value="cruise" className="bg-[#0A0E14] text-[#E8EDF2]">Cruise Leg {idx + 1}</option>
                      <option value="loiter" className="bg-[#0A0E14] text-[#E8EDF2]">Loiter Leg {idx + 1}</option>
                    </select>
                    <span className="pointer-events-none absolute right-1.5 top-1/2 -translate-y-1/2 text-[7px] text-[#FFB454]">▾</span>
                  </div>
                  <button
                    onClick={() => removeLeg(leg.id)}
                    disabled={loading || legs.length <= 1}
                    className="text-[9px] text-red-400 hover:text-red-300 disabled:opacity-30 px-1"
                  >
                    ✕ Remove
                  </button>
                </div>

                <div className="mb-3">
                  <div className="flex justify-between text-[10px] mb-1">
                    <span className="text-[#5C6773]">Altitude</span>
                    <span className="font-mono text-[#FFB454] font-bold bg-[#0A0E14] px-1.5 rounded border border-[#1F2733]">{leg.altitude_m}m</span>
                  </div>
                  <input type="range" min={baseElevationM + 300} max="10000" step="100" value={leg.altitude_m}
                    onChange={(e) => updateLeg(leg.id, { altitude_m: parseInt(e.target.value) })} disabled={loading}
                    className="w-full h-1 bg-slate-800 rounded appearance-none cursor-pointer accent-[#FFB454] disabled:opacity-40" />
                </div>

                <div className="mb-3">
                  <div className="flex justify-between text-[10px] mb-1">
                    <span className="text-[#5C6773]">Speed</span>
                    <span className="font-mono text-[#FFB454] font-bold bg-[#0A0E14] px-1.5 rounded border border-[#1F2733]">{leg.speed_kmh} km/h</span>
                  </div>
                  <input type="range" min="100" max="350" step="5" value={leg.speed_kmh}
                    onChange={(e) => updateLeg(leg.id, { speed_kmh: parseInt(e.target.value) })} disabled={loading}
                    className="w-full h-1 bg-slate-800 rounded appearance-none cursor-pointer accent-[#FFB454] disabled:opacity-40" />
                </div>

                {leg.role === 'cruise' ? (
                  <>
                    <div className="mb-3">
                      <div className="flex justify-between text-[10px] mb-1">
                        <span className="text-[#5C6773]">Distance</span>
                        <span className="font-mono text-[#FFB454] font-bold bg-white/10 px-1.5 rounded">{leg.distance_km} km</span>
                      </div>
                      <input type="range" min="50" max="1000" step="25" value={leg.distance_km}
                        onChange={(e) => updateLeg(leg.id, { distance_km: parseInt(e.target.value) })} disabled={loading}
                        className="w-full h-1 bg-slate-800 rounded appearance-none cursor-pointer accent-[#FFB454] disabled:opacity-40" />
                    </div>
                    <div className="mb-1">
                      <div className="flex justify-between text-[10px] mb-1">
                        <span className="text-[#5C6773]">Wind</span>
                        <span className="font-mono text-[#FFB454] font-bold bg-white/10 px-1.5 rounded">
                          {leg.windKmh === 0 ? 'Calm' : `${Math.abs(leg.windKmh)} km/h ${leg.windKmh > 0 ? 'Headwind' : 'Tailwind'}`}
                        </span>
                      </div>
                      <input type="range" min="-60" max="60" step="5" value={leg.windKmh}
                        onChange={(e) => updateLeg(leg.id, { windKmh: parseInt(e.target.value) })} disabled={loading}
                        className="w-full h-1 bg-slate-800 rounded appearance-none cursor-pointer accent-[#FFB454] disabled:opacity-40" />
                    </div>
                  </>
                ) : (
                  <div className="mb-1">
                    <div className="flex justify-between text-[10px] mb-1">
                      <span className="text-[#5C6773]">Duration</span>
                      <span className="font-mono text-[#FFB454] font-bold bg-white/10 px-1.5 rounded">{leg.duration_min} min</span>
                    </div>
                    <input type="range" min="10" max="300" step="10" value={leg.duration_min}
                      onChange={(e) => updateLeg(leg.id, { duration_min: parseInt(e.target.value) })} disabled={loading}
                      className="w-full h-1 bg-slate-800 rounded appearance-none cursor-pointer accent-[#FFB454] disabled:opacity-40" />
                  </div>
                )}
              </div>
            ))}

            <div className="flex gap-1.5 mt-3 mb-3">
              <button
                onClick={() => addLeg('cruise')} disabled={loading}
                className="flex-1 text-[9px] py-1 rounded border border-[#1F2733] text-cyan-300 hover:bg-cyan-900/20 disabled:opacity-40"
              >
                + Cruise Leg
              </button>
              <button
                onClick={() => addLeg('loiter')} disabled={loading}
                className="flex-1 text-[9px] py-1 rounded border border-[#1F2733] text-violet-300 hover:bg-violet-900/20 disabled:opacity-40"
              >
                + Loiter Leg
              </button>
            </div>

            {legWarnings.length > 0 && (
              <div className="mb-3 bg-amber-950/30 border border-amber-800/40 rounded-md p-2">
                {legWarnings.map((w, i) => (
                  <p key={i} className="text-[9px] text-amber-300 leading-snug mb-1 last:mb-0">⚠ {w}</p>
                ))}
              </div>
            )}

            <div className="mb-3 pt-3 border-t border-[#1F2733]">
              <div className="flex justify-between text-[10px] mb-1">
                <span className="text-[#5C6773]">Base Elevation</span>
                <span className="font-mono text-[#FFB454] font-bold bg-[#0A0E14] px-1.5 rounded border border-[#1F2733]">{baseElevationM}m</span>
              </div>
              <input type="range" min="0" max="5000" step="100" value={baseElevationM}
                onChange={(e) => setBaseElevationM(parseInt(e.target.value))} disabled={loading}
                className="w-full h-1 bg-slate-800 rounded appearance-none cursor-pointer accent-[#FFB454] disabled:opacity-40" />
            </div>

            <div className="mb-3">
              <div className="flex justify-between text-[10px] mb-1">
                <span className="text-[#5C6773]">Ambient Temp</span>
                <span className="font-mono text-[#FFB454] font-bold bg-[#0A0E14] px-1.5 rounded border border-[#1F2733]">{ambientTempC}°C</span>
              </div>
              <input type="range" min="-30" max="45" step="1" value={ambientTempC}
                onChange={(e) => setAmbientTempC(parseInt(e.target.value))} disabled={loading}
                className="w-full h-1 bg-slate-800 rounded appearance-none cursor-pointer accent-[#FFB454] disabled:opacity-40" />
            </div>

            <div className="mb-3">
              <div className="flex justify-between text-[10px] mb-1">
                <span className="text-[#5C6773]">Turbulence</span>
                <span className="font-mono text-[#FFB454] font-bold bg-[#0A0E14] px-1.5 rounded border border-[#1F2733]">{turbulenceLevel.toFixed(1)}</span>
              </div>
              <input type="range" min="0" max="1" step="0.1" value={turbulenceLevel}
                onChange={(e) => setTurbulenceLevel(parseFloat(e.target.value))} disabled={loading}
                className="w-full h-1 bg-slate-800 rounded appearance-none cursor-pointer accent-[#FFB454] disabled:opacity-40" />
            </div>

            <div className="mb-3">
              <div className="flex justify-between text-[10px] mb-1">
                <span className="text-[#5C6773]">Payload Mass</span>
                <span className="font-mono text-[#FFB454] font-bold bg-[#0A0E14] px-1.5 rounded border border-[#1F2733]">{payloadWeight} kg</span>
              </div>
              <input type="range" min="100" max="300" step="5" value={payloadWeight}
                onChange={(e) => setPayloadWeight(parseInt(e.target.value))} disabled={loading}
                className="w-full h-1 bg-slate-800 rounded appearance-none cursor-pointer accent-[#FFB454] disabled:opacity-40" />
            </div>

            <div className="mb-3">
              <div className="flex justify-between text-[10px] mb-1">
                <span className="text-[#5C6773]">Initial Fuel Load</span>
                <span className="font-mono text-orange-400 font-bold bg-[#0A0E14] px-1.5 rounded border border-[#1F2733]">{Math.round(initialFuelFraction * 100)}%</span>
              </div>
              <input type="range" min="0.1" max="1.0" step="0.05" value={initialFuelFraction}
                onChange={(e) => setInitialFuelFraction(parseFloat(e.target.value))} disabled={loading}
                className="w-full h-1 bg-slate-800 rounded appearance-none cursor-pointer accent-orange-500 disabled:opacity-40" />
            </div>

            <div className="flex items-center gap-2 mb-2">
              <input type="checkbox" id="silentLoiterToggle" checked={silentLoiterMode}
                onChange={(e) => setSilentLoiterMode(e.target.checked)} disabled={loading}
                className="w-3.5 h-3.5 rounded accent-[#FFB454] cursor-pointer disabled:opacity-40" />
              <label htmlFor="silentLoiterToggle" className="text-[10px] text-[#5C6773] cursor-pointer">Silent Loiter Mode (stealth)</label>
            </div>

            <button
              onClick={() => setShowAdvanced(!showAdvanced)}
              className="text-[9px] text-[#5C6773] hover:text-[#E8EDF2] mb-3 underline"
            >
              {showAdvanced ? '− Hide' : '+ Show'} advanced variables
            </button>

            {showAdvanced && (
              <div className="mb-4 space-y-2 pt-2 border-t border-[#1F2733]">
                <label className="flex flex-col gap-0.5">
                  <span className="text-[10px] text-[#5C6773]">Battery Chemistry</span>
                  <div className="relative">
                    <select
                      value={batteryChemistry} disabled={loading}
                      onChange={(e) => setBatteryChemistry(e.target.value)}
                      className="appearance-none w-full bg-[#0A0E14] border border-[#1F2733] rounded text-[10px] pl-2 pr-6 py-1.5 text-[#E8EDF2] cursor-pointer disabled:opacity-40 hover:border-[#FFB454]/40 focus:outline-none focus:border-[#FFB454]/60"
                    >
                      <option value="Li-NCA" className="bg-[#0A0E14] text-[#E8EDF2]">Li-NCA (250 Wh/kg, 3C/5C)</option>
                      <option value="Li-LFP" className="bg-[#0A0E14] text-[#E8EDF2]">Li-LFP (160 Wh/kg, 2.5C/4C, cold-tolerant)</option>
                    </select>
                    <span className="pointer-events-none absolute right-2 top-1/2 -translate-y-1/2 text-[8px] text-[#FFB454]">▾</span>
                  </div>
                </label>
                <div className="flex items-center gap-2">
                  <input type="checkbox" id="optPsrToggle" checked={optimizePowerSplit}
                    onChange={(e) => setOptimizePowerSplit(e.target.checked)} disabled={loading}
                    className="w-3.5 h-3.5 rounded accent-[#FFB454] cursor-pointer disabled:opacity-40" />
                  <label htmlFor="optPsrToggle" className="text-[10px] text-[#5C6773] cursor-pointer">GA-search power-split policy (slower)</label>
                </div>
              </div>
            )}

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
            <div className="bg-[#12161F] border border-[#1F2733] rounded-lg shadow-md m-3 p-5 panel-enter" style={{ animationDelay: '50ms' }}>
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

          {/* ── Mission Profile ─── */}
          {phaseSegments && specs && !loading && (
            <div className="bg-[#12161F] border border-[#1F2733] rounded-lg shadow-md m-3 p-5 panel-enter" style={{ animationDelay: '100ms' }}>
              <h2 className="text-[9px] font-bold text-[#5C6773] uppercase tracking-[0.15em] mb-3 flex items-center gap-1.5">
                <span>🗺️</span> Mission Profile
              </h2>

              <div className="flex items-center justify-between mb-3 text-[10px]">
                <span className="text-[#5C6773]">On-Station Requirement</span>
                <span className={`font-bold font-mono ${onStationMet ? 'text-emerald-400' : 'text-red-400'}`}>
                  {specs.on_station_min_achieved.toFixed(0)}/{specs.on_station_min_required.toFixed(0)}min {onStationMet ? '✓ MET' : '✗ CUT SHORT'}
                </span>
              </div>

              <div className="relative pl-4">
                {/* Vertical timeline line */}
                <div className="absolute left-1.5 top-1.5 bottom-1.5 w-px bg-gradient-to-b from-emerald-500/60 via-slate-700/40 to-transparent" />
                {phaseSegments
                  .filter((s) => s.phase !== 'completed')
                  .map((seg, idx) => {
                    const durSec = seg.end - seg.start;
                    const durMin = durSec / 60;
                    const fraction = Math.min(durSec / totalMissionTime, 1);
                    return (
                      <div key={`${seg.phase}-${idx}`} className="mb-2.5 last:mb-0">
                        {/* Dot + label row */}
                        <div className="flex items-center justify-between mb-1" style={{ animationDelay: `${idx * 40}ms` }}>
                          <div className="flex items-center gap-2">
                            <span className="absolute left-0.5 w-2 h-2 rounded-full bg-emerald-500 border border-emerald-300 shadow-[0_0_6px_rgba(255,180,84,0.7)]"
                              style={{ marginTop: 0 }} />
                            <span className="capitalize font-medium text-[10px] text-slate-300">{seg.phase}</span>
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

              <div className="flex justify-between text-[9px] text-[#5C6773] mt-3 pt-2 border-t border-[#1F2733]">
                <span>Reserve: {specs.reserve_fuel_equivalent_min.toFixed(0)}min fuel / {specs.reserve_battery_equivalent_min.toFixed(0)}min batt</span>
                <span className={specs.engine_out_survivable ? 'text-emerald-400' : 'text-amber-400'}>
                  {specs.engine_out_survivable ? '✓' : '⚠'} engine-out
                </span>
              </div>
            </div>
          )}

          {/* ── Weight Budget ─── */}
          {weightBreakdown && !loading && (
            <div className="bg-[#12161F] border border-[#1F2733] rounded-lg shadow-md m-3 p-5 panel-enter" style={{ animationDelay: '150ms' }}>
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
            <div className="bg-[#12161F] border border-[#1F2733] rounded-lg shadow-md m-3 p-5 panel-enter" style={{ animationDelay: '200ms' }}>
              <h2 className="text-[9px] font-bold text-[#5C6773] uppercase tracking-[0.15em] mb-2 flex items-center gap-1.5">
                <span>🔬</span> System Constants
              </h2>
              <div className="space-y-1 text-[10px]">
                <div className="flex justify-between"><span className="text-[#5C6773]">Drag Polar</span><span className="font-mono text-slate-300">Oswald AR=16.07</span></div>
                <div className="flex justify-between"><span className="text-[#5C6773]">L/D Max</span><span className="font-mono text-slate-300">{specs.l_over_d_max.toFixed(1)}</span></div>
                <div className="flex justify-between"><span className="text-[#5C6773]">SFC</span><span className="font-mono text-slate-300">{currentPoint && currentPoint.sfc !== undefined ? `${currentPoint.sfc.toFixed(3)} kg/kWh` : '0.380 kg/kWh'}</span></div>
                <div className="flex justify-between"><span className="text-[#5C6773]">C-Rate Limit</span><span className="font-mono text-yellow-400">3C/5C</span></div>
                <div className="flex justify-between"><span className="text-[#5C6773]">Motor η</span><span className="font-mono text-slate-300">96%</span></div>
                <div className="flex justify-between"><span className="text-[#5C6773]">Chemistry</span><span className="font-mono text-slate-300">{specs.battery_chemistry}</span></div>
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
                    <p className="text-[10px] text-[#5C6773] mt-1 font-mono">Sizing propulsion architecture against your mission profile</p>
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
              <div
                className="w-[620px] flex-shrink-0 border-l border-white/10 flex flex-col overflow-hidden shadow-[inset_1px_0_0_rgba(255,255,255,0.04)]"
                style={{
                  background: 'rgba(10,14,20,0.78)',
                  backdropFilter: 'blur(28px)',
                  WebkitBackdropFilter: 'blur(28px)',
                }}
              >
                <div className="h-7 flex-shrink-0 flex items-center px-2 border-b border-[#1F2733] bg-white/5">
                  <h3 className="text-[9px] font-bold text-[#E8EDF2] uppercase tracking-[0.15em]">Propulsion Status Matrix — kW</h3>
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
