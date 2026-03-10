/// <reference types="vite/client" />
import { RefObject, useState } from "react";
import { Camera, Loader2, RefreshCw } from "lucide-react";
import { useIngredients } from "../../services/IngredientContext";
import { DetectionSummary } from "../inventory_management/DetectionSummary";

interface CameraViewProps {
    videoRef: RefObject<HTMLVideoElement | null>;
}

export function CameraView({ videoRef }: CameraViewProps) {
    const { addItem, scannedItems, tempDetections, clearTempDetections } = useIngredients();
    const [isScanning, setIsScanning] = useState(false); // 是否正在進行 AI 掃描
    const [showFrames, setShowFrames] = useState(false); // 是否顯示 AI 識別框
    const [currentBoxes, setCurrentBoxes] = useState<any[]>([]); // 存放當前畫面偵測到的物體座標與資訊

    const handleScan = async () => {
        if (!videoRef.current) return;
        setIsScanning(true);
        setShowFrames(false);
        setCurrentBoxes([]); // 開始新掃描前先清空舊的識別框

        // --- 手動調整參數：模擬延遲感 (毫秒) ---
        // 增加 AI 運算的真實體感預留時間
        await new Promise(resolve => setTimeout(resolve, 1500));

        try {
            const canvas = document.createElement("canvas");
            canvas.width = videoRef.current.videoWidth;
            canvas.height = videoRef.current.videoHeight;
            const ctx = canvas.getContext("2d");
            if (ctx) {
                ctx.drawImage(videoRef.current, 0, 0);
                const imageData = canvas.toDataURL("image/jpeg");
                const apiUrl = import.meta.env.VITE_DETECTION_API_URL || "http://localhost:8000";

                const response = await fetch(`${apiUrl}/detect`, {
                    method: "POST",
                    headers: { "Content-Type": "application/json", "bypass-tunnel-reminder": "true" },
                    body: JSON.stringify({ image: imageData })
                });

                const data = await response.json();
                if (data.detections?.length > 0) {
                    // --- 手動調整參數：AI 信心度過濾門檻 (0.0 ~ 1.0) ---
                    // 數值越高越精準，但太高會掃不到東西；數值越低越靈敏，但容易有幽靈辨識
                    const CONFIDENCE_THRESHOLD = 0.2;

                    const validDetections = data.detections.filter((det: any) => (det.confidence || 0) > CONFIDENCE_THRESHOLD);

                    setCurrentBoxes(validDetections);
                    validDetections.forEach((det: any) => addItem({
                        name: det.name,
                        quantity: 1,
                        category: det.category || "其他",
                        confidence: det.confidence,
                        isSpoiled: det.isSpoiled,
                        box: det.box
                    }));
                }
            }
        } catch (error) {
            console.warn("辨識服務連接失敗:", error);
        } finally {
            setIsScanning(false);
            setShowFrames(true);
        }
    };

    const handleClear = () => {
        clearTempDetections();
        setCurrentBoxes([]);
    };

    return (
        <div className="flex flex-col items-center w-full max-w-sm">
            <div className="relative w-full">
                {/* AI Vision Active Badge */}
                <div className={`absolute top-4 left-1/2 transform -translate-x-1/2 z-20 bg-[#0f2e24]/80 backdrop-blur-md border ${isScanning ? 'border-amber-400' : 'border-[#00ff88]'} rounded-full px-4 py-1.5 flex items-center gap-2 shadow-[0_0_15px_rgba(0,255,136,0.3)] transition-colors duration-500`}>
                    <div className={`w-2 h-2 rounded-full ${isScanning ? 'bg-amber-400 shadow-[0_0_8px_#fbbf24]' : 'bg-[#00ff88] shadow-[0_0_8px_#00ff88]'} animate-pulse`} />
                    <span className={`text-[10px] font-black tracking-widest ${isScanning ? 'text-amber-400' : 'text-[#00ff88]'} uppercase`}>
                        {isScanning ? "Analyzing..." : "核心感測運作中"}
                    </span>
                </div>

                {/* Camera View */}
                <div className="relative aspect-[3/4] bg-[#1a4d3d] rounded-[2.5rem] overflow-hidden border-4 border-[#1a4d3d] shadow-2xl">
                    <video
                        ref={videoRef}
                        autoPlay
                        playsInline
                        muted
                        className="absolute inset-0 w-full h-full object-cover"
                    />

                    {/* Bounding Box Overlay */}
                    <div className="absolute inset-0 z-10 pointer-events-none">
                        {currentBoxes.map((boxData, idx) => boxData.box && (
                            <div
                                key={`box-${idx}`}
                                className="absolute"
                                style={{
                                    left: `${boxData.box[0] * 100}%`,
                                    top: `${boxData.box[1] * 100}%`,
                                    width: `${(boxData.box[2] - boxData.box[0]) * 100}%`,
                                    height: `${(boxData.box[3] - boxData.box[1]) * 100}%`,
                                    borderColor: boxData.isSpoiled ? '#ff4d4d' : '#00ff88',
                                    borderWidth: '2px',
                                    borderStyle: 'solid',
                                    borderRadius: '8px'
                                }}
                            >
                                <div className={`absolute -top-6 left-0 px-2 py-0.5 rounded-t-md text-[8px] font-black uppercase whitespace-nowrap ${boxData.isSpoiled ? 'bg-red-500 text-white' : 'bg-[#00ff88] text-[#0f2e24]'}`}>
                                    {boxData.isSpoiled ? 'BAD' : 'GOOD'} | {boxData.name} | {Math.round((boxData.confidence || 0) * 100)}%
                                </div>
                            </div>
                        ))}
                    </div>

                    <div className="absolute inset-0 bg-gradient-to-b from-[#0f2e24]/40 to-transparent pointer-events-none" />

                    {isScanning && (
                        <div className="absolute inset-0 bg-[#00ff88]/5 flex flex-col items-center justify-center">
                            <div className="w-full h-[2px] bg-amber-400 shadow-[0_0_15px_#fbbf24] absolute top-0 animate-[scan_2s_ease-in-out_infinite]" />
                        </div>
                    )}

                    <div className={`absolute top-8 left-8 w-10 h-10 border-l-4 border-t-4 ${isScanning ? 'border-amber-400' : 'border-[#00ff88]'} rounded-tl-xl opacity-80 transition-colors duration-500`} />
                    <div className={`absolute top-8 right-8 w-10 h-10 border-r-4 border-t-4 ${isScanning ? 'border-amber-400' : 'border-[#00ff88]'} rounded-tr-xl opacity-80 transition-colors duration-500`} />
                    <div className={`absolute bottom-8 left-8 w-10 h-10 border-l-4 border-b-4 ${isScanning ? 'border-amber-400' : 'border-[#00ff88]'} rounded-bl-xl opacity-80 transition-colors duration-500`} />
                    <div className={`absolute bottom-8 right-8 w-10 h-10 border-r-4 border-b-4 ${isScanning ? 'border-amber-400' : 'border-[#00ff88]'} rounded-br-xl opacity-80 transition-colors duration-500`} />
                </div>
            </div>

            {tempDetections.length > 0 && (
                <div className="w-full flex justify-end px-2 mb-2 mt-4">
                    <button
                        onClick={handleClear}
                        className="flex items-center gap-2 px-3 py-1.5 bg-red-500/10 text-red-500 rounded-lg border border-red-500/20 hover:bg-red-500 hover:text-white transition-all text-[10px] font-black tracking-widest uppercase"
                    >
                        <RefreshCw size={12} />
                        重新整理畫面
                    </button>
                </div>
            )}

            {/* 即時辨識清單：放置在相機預覽框下方 */}
            <DetectionSummary readOnly={true} />

            {/* Scan Button Group */}
            <div className="w-full mt-8 space-y-3 px-2">
                <button
                    onClick={handleScan}
                    disabled={isScanning}
                    className="w-full bg-[#00ff88] text-[#0f2e24] py-4 rounded-2xl font-black text-lg flex items-center justify-center gap-3 hover:bg-[#00dd77] transition-all active:scale-[0.98] shadow-[0_8px_20px_rgba(0,255,136,0.3)] disabled:opacity-50"
                >
                    {isScanning ? <Loader2 size={24} className="animate-spin" /> : <Camera size={24} strokeWidth={3} />}
                    {isScanning ? "NEURAL LINKING..." : "開始掃描食材"}
                </button>
            </div>
        </div>
    );
}
