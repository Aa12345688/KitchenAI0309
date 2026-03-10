import { Trash2, Plus, Minus } from "lucide-react";

interface DetectionRowProps {
    item: any;
    onUpdate: (id: string, delta: number) => void;
    onRemove: (id: string) => void;
}

export function DetectionRow({ item, onUpdate, onRemove }: DetectionRowProps) {
    return (
        <div className="bg-[#1a4d3d]/30 backdrop-blur-xl rounded-[1.5rem] p-4 border border-white/5 hover:border-[#00ff88]/20 transition-all group active:scale-[0.98]">
            <div className="flex items-center gap-4">
                {/* Placeholder for item image if available */}
                <div className="w-12 h-12 rounded-xl bg-[#0f2e24] border border-white/5 flex items-center justify-center flex-shrink-0 shadow-inner group-hover:scale-110 transition-transform">
                    <div className="w-2 h-2 rounded-full bg-[#00ff88] shadow-[0_0_8px_#00ff88]" />
                </div>

                <div className="flex-1 min-w-0">
                    <h3 className="font-black text-white text-sm tracking-tight mb-1 uppercase truncate group-hover:text-[#00ff88] transition-colors leading-tight">
                        {item.name}
                    </h3>
                    <div className="flex items-center gap-2">
                        <span className="text-[9px] font-black text-[#00ff88]/50 uppercase tracking-widest">
                            已驗證節點
                        </span>
                    </div>
                </div>

                <div className="flex items-center gap-3">
                    <div className="flex items-center bg-[#0f2e24] rounded-full p-1 border border-white/10 shadow-inner">
                        <button
                            onClick={() => onUpdate(item.id, -1)}
                            className="w-8 h-8 flex items-center justify-center hover:bg-[#00ff88] hover:text-[#0f2e24] rounded-full transition-all text-gray-500"
                        >
                            <Minus size={14} strokeWidth={3} />
                        </button>
                        <span className="w-8 text-center font-black text-[#00ff88] text-sm tabular-nums">
                            {item.quantity}
                        </span>
                        <button
                            onClick={() => onUpdate(item.id, 1)}
                            className="w-8 h-8 flex items-center justify-center hover:bg-[#00ff88] hover:text-[#0f2e24] rounded-full transition-all text-gray-500"
                        >
                            <Plus size={14} strokeWidth={3} />
                        </button>
                    </div>
                    <button
                        onClick={() => onRemove(item.id)}
                        className="w-10 h-10 rounded-xl bg-red-500/10 text-red-500 border border-red-500/10 flex items-center justify-center hover:bg-red-500 hover:text-white transition-all shadow-lg active:scale-90"
                    >
                        <Trash2 size={16} strokeWidth={2.5} />
                    </button>
                </div>
            </div>
        </div>
    );
}
