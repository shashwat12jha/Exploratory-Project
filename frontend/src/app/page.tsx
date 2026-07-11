"use client";

import React, { useState } from "react";
import { 
  UploadCloud, 
  PlayCircle, 
  BarChart3, 
  Activity, 
  AlertTriangle,
  Clock,
  Settings
} from "lucide-react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend
} from "recharts";

// Mock data for the chart
const mockChartData = [
  { time: "00:00", mttc: 4, pet: 2 },
  { time: "00:05", mttc: 7, pet: 3 },
  { time: "00:10", mttc: 2, pet: 5 },
  { time: "00:15", mttc: 9, pet: 4 },
  { time: "00:20", mttc: 5, pet: 7 },
  { time: "00:25", mttc: 3, pet: 2 },
];

export default function Dashboard() {
  const [file, setFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [status, setStatus] = useState<"idle" | "uploading" | "processing" | "completed">("idle");

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
    }
  };

  const handleUpload = () => {
    if (!file) return;
    setUploading(true);
    setStatus("uploading");
    
    // Simulate API Call
    setTimeout(() => {
      setStatus("processing");
      setTimeout(() => {
        setStatus("completed");
        setUploading(false);
      }, 3000);
    }, 1500);
  };

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col font-sans text-gray-900">
      
      {/* Header */}
      <header className="bg-white border-b border-gray-200 px-8 py-4 flex items-center justify-between sticky top-0 z-10 shadow-sm">
        <div className="flex items-center gap-3">
          <div className="bg-blue-600 p-2 rounded-lg">
            <Activity className="text-white w-6 h-6" />
          </div>
          <div>
            <h1 className="text-xl font-bold text-gray-900">Traffic Safety Analytics</h1>
            <p className="text-xs text-gray-500 font-medium">Urban Streets Surveillance System</p>
          </div>
        </div>
        <div className="flex items-center gap-4">
          <button className="text-gray-500 hover:text-gray-900 transition">
            <Settings className="w-5 h-5" />
          </button>
          <div className="w-9 h-9 rounded-full bg-blue-100 flex items-center justify-center text-blue-700 font-bold border border-blue-200">
            SJ
          </div>
        </div>
      </header>

      {/* Main Content Layout */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-8 grid grid-cols-1 lg:grid-cols-3 gap-8">
        
        {/* Left Column: Upload & Video */}
        <div className="lg:col-span-2 space-y-8">
          
          {/* Upload Section */}
          <section className="bg-white p-6 rounded-xl border border-gray-200 shadow-sm">
            <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
              <UploadCloud className="w-5 h-5 text-blue-600" />
              Upload Traffic Video
            </h2>
            <div className="border-2 border-dashed border-gray-300 rounded-xl p-8 text-center bg-gray-50 hover:bg-gray-100 transition">
              <input 
                type="file" 
                id="video-upload" 
                className="hidden" 
                accept="video/mp4,video/x-m4v,video/*"
                onChange={handleFileChange}
              />
              <label htmlFor="video-upload" className="cursor-pointer flex flex-col items-center">
                <div className="w-16 h-16 bg-blue-50 text-blue-600 rounded-full flex items-center justify-center mb-4">
                  <UploadCloud className="w-8 h-8" />
                </div>
                <span className="text-gray-700 font-medium text-lg">
                  {file ? file.name : "Click or drag video to upload"}
                </span>
                <span className="text-gray-400 text-sm mt-1">MP4, AVI, or MOV (Max 500MB)</span>
              </label>
              
              {file && status === "idle" && (
                <button 
                  onClick={handleUpload}
                  className="mt-6 bg-blue-600 text-white px-6 py-2.5 rounded-lg font-medium shadow hover:bg-blue-700 transition w-full max-w-xs"
                >
                  Analyze Video
                </button>
              )}
            </div>

            {/* Status Progress Bar */}
            {status !== "idle" && (
              <div className="mt-6">
                <div className="flex justify-between text-sm font-medium text-gray-700 mb-2">
                  <span>
                    {status === "uploading" ? "Uploading video..." : 
                     status === "processing" ? "Analyzing trajectories and computing MTTC/PET..." : 
                     "Analysis Complete"}
                  </span>
                  <span>{status === "completed" ? "100%" : "Processing..."}</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2.5">
                  <div 
                    className={`bg-blue-600 h-2.5 rounded-full transition-all duration-1000 ${
                      status === "uploading" ? "w-1/3" : 
                      status === "processing" ? "w-2/3" : "w-full"
                    }`}
                  ></div>
                </div>
              </div>
            )}
          </section>

          {/* Video Player Section */}
          <section className="bg-white p-6 rounded-xl border border-gray-200 shadow-sm">
            <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
              <PlayCircle className="w-5 h-5 text-blue-600" />
              Processed Output
            </h2>
            <div className="aspect-video bg-gray-900 rounded-xl flex items-center justify-center overflow-hidden relative shadow-inner">
              {status === "completed" ? (
                // Placeholder for actual processed video
                <div className="w-full h-full flex flex-col items-center justify-center text-gray-400 bg-gray-800">
                  <PlayCircle className="w-16 h-16 text-gray-500 mb-4 opacity-50" />
                  <p>Video processing complete.</p>
                  <p className="text-sm">In production, the annotated video will stream here.</p>
                </div>
              ) : (
                <span className="text-gray-500">No video processed yet.</span>
              )}
            </div>
          </section>
        </div>

        {/* Right Column: Analytics */}
        <div className="space-y-8">
          
          {/* Key Metrics */}
          <div className="grid grid-cols-2 gap-4">
            <div className="bg-white p-5 rounded-xl border border-gray-200 shadow-sm flex flex-col">
              <span className="text-gray-500 text-sm font-medium flex items-center gap-1">
                <AlertTriangle className="w-4 h-4 text-orange-500" /> Rear-End Risks
              </span>
              <span className="text-3xl font-bold text-gray-900 mt-2">
                {status === "completed" ? "24" : "--"}
              </span>
              <span className="text-xs text-gray-400 mt-1">MTTC &lt; 1.5s</span>
            </div>
            
            <div className="bg-white p-5 rounded-xl border border-gray-200 shadow-sm flex flex-col">
              <span className="text-gray-500 text-sm font-medium flex items-center gap-1">
                <Activity className="w-4 h-4 text-red-500" /> Crossing Risks
              </span>
              <span className="text-3xl font-bold text-gray-900 mt-2">
                {status === "completed" ? "12" : "--"}
              </span>
              <span className="text-xs text-gray-400 mt-1">PET &lt; 1.0s</span>
            </div>

            <div className="bg-white p-5 rounded-xl border border-gray-200 shadow-sm flex flex-col col-span-2">
              <span className="text-gray-500 text-sm font-medium flex items-center gap-1">
                <Clock className="w-4 h-4 text-blue-500" /> Total Vehicles Tracked
              </span>
              <span className="text-3xl font-bold text-gray-900 mt-2">
                {status === "completed" ? "1,402" : "--"}
              </span>
            </div>
          </div>

          {/* Charts */}
          <section className="bg-white p-6 rounded-xl border border-gray-200 shadow-sm">
            <h2 className="text-lg font-semibold mb-6 flex items-center gap-2">
              <BarChart3 className="w-5 h-5 text-blue-600" />
              Conflict Frequency
            </h2>
            <div className="h-64 w-full">
              {status === "completed" ? (
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={mockChartData} margin={{ top: 5, right: 10, left: -20, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#E5E7EB" />
                    <XAxis dataKey="time" axisLine={false} tickLine={false} tick={{ fontSize: 12, fill: '#6B7280' }} dy={10} />
                    <YAxis axisLine={false} tickLine={false} tick={{ fontSize: 12, fill: '#6B7280' }} />
                    <Tooltip 
                      contentStyle={{ borderRadius: '8px', border: 'none', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)' }}
                    />
                    <Legend iconType="circle" wrapperStyle={{ fontSize: '12px', paddingTop: '10px' }} />
                    <Line type="monotone" dataKey="mttc" name="Rear-End (MTTC)" stroke="#f97316" strokeWidth={3} dot={{ r: 4 }} activeDot={{ r: 6 }} />
                    <Line type="monotone" dataKey="pet" name="Crossing (PET)" stroke="#ef4444" strokeWidth={3} dot={{ r: 4 }} activeDot={{ r: 6 }} />
                  </LineChart>
                </ResponsiveContainer>
              ) : (
                <div className="w-full h-full flex items-center justify-center text-gray-400 bg-gray-50 rounded-lg border border-dashed border-gray-200">
                  Data will appear after processing
                </div>
              )}
            </div>
          </section>

          {/* Heatmap Placeholder */}
          <section className="bg-white p-6 rounded-xl border border-gray-200 shadow-sm">
            <h2 className="text-lg font-semibold mb-4">Risk Heatmap</h2>
            <div className="aspect-video bg-gray-100 rounded-lg flex items-center justify-center relative overflow-hidden border border-gray-200">
              {status === "completed" ? (
                <div className="absolute inset-0 bg-gradient-to-tr from-blue-100 via-yellow-100 to-red-100 opacity-60"></div>
              ) : null}
              <span className="text-gray-400 z-10 bg-white/80 px-3 py-1 rounded shadow-sm text-sm">
                {status === "completed" ? "Heatmap Generated" : "Awaiting Data"}
              </span>
            </div>
          </section>

        </div>
      </main>
    </div>
  );
}
