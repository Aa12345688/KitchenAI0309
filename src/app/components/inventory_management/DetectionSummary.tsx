import { ChevronRight, Plus, Minus, X, Info } from "lucide-react";
import { useIngredients } from "../../services/IngredientContext";
import { useNavigate } from "react-router";

export function DetectionSummary({ readOnly = false }: { readOnly?: boolean }) {
    const { tempDetections, updateQuantity, removeItem, clearTempDetections, selectedIds, toggleSelection } = useIngredients();
    const navigate = useNavigate();

    if (tempDetections.length === 0) return null;

    return (
        <div className="w-full mt-6 px-2 animate-in fade-in slide-in-from-top-4 duration-500">
            <div className="bg-[#0f2e24]/60 backdrop-blur-xl border border-[#00ff88]/20 rounded-[2.5rem] p-5 shadow-2xl overflow-hidden">
                <div className="flex items-center justify-between mb-4 px-2">
                    <div className="flex items-center gap-2">
                        <div className="w-2 h-2 rounded-full bg-[#00ff88] animate-pulse" />
                        <h3 className="text-[10px] font-black tracking-widest text-[#00ff88] uppercase">
                            辨識清單 ({tempDetections.length})
                            {selectedIds.length > 0 && <span className="ml-2 text-white/40">已選 {selectedIds.length}</span>}
                        </h3>
                    </div>
                </div>

                <div className="space-y-2 max-h-48 overflow-y-auto no-scrollbar pr-1">
                    {tempDetections.map((item) => (
                        <div
                            key={item.id}
                            className={`bg-[#1a4d3d]/40 border ${selectedIds.includes(item.id) ? 'border-[#00ff88]/40 shadow-[0_0_10px_rgba(0,255,136,0.1)]' : 'border-white/5'} rounded-2xl px-4 py-3 flex items-center justify-between group animate-in zoom-in duration-300 transition-all`}
                        >
                            <div className="flex items-center gap-4 flex-1 min-w-0">
                                {/* Checkbox */}
                                {!readOnly && (
                                    <button
                                        onClick={() => toggleSelection(item.id)}
                                        className={`w-5 h-5 rounded-md border-2 flex items-center justify-center transition-all ${selectedIds.includes(item.id) ? 'bg-[#00ff88] border-[#00ff88]' : 'bg-transparent border-white/20'}`}
                                    >
                                        {selectedIds.includes(item.id) && <div className="w-2.5 h-2.5 bg-[#0f2e24] rounded-sm" />}
                                    </button>
                                )}

                                <div className="flex-1 min-w-0">
                                    <div className="text-xs font-black text-white uppercase truncate">{item.name}</div>
                                    <div className="text-[9px] font-bold text-gray-500 uppercase">{item.category}</div>
                                </div>
                            </div>

                            {readOnly ? (
                                <div className="flex items-center px-4 py-1.5 bg-[#0f2e24]/80 rounded-full border border-white/10">
                                    <span className="text-[10px] font-black text-[#00ff88] uppercase tracking-wider mr-2 opacity-50">Count</span>
                                    <span className="text-xs font-black text-[#00ff88]">{item.quantity}</span>
                                </div>
                            ) : (
                                <div className="flex items-center gap-3">
                                    <div className="flex items-center bg-[#0f2e24]/80 rounded-full p-1 border border-white/10 scale-90">
                                        <button
                                            onClick={() => updateQuantity(item.id, -1)}
                                            className="w-8 h-8 flex items-center justify-center text-gray-400 hover:text-[#00ff88] transition-colors"
                                        >
                                            <Minus size={14} strokeWidth={3} />
                                        </button>
                                        <span className="w-6 text-center font-black text-[#00ff88] text-xs">{item.quantity}</span>
                                        <button
                                            onClick={() => updateQuantity(item.id, 1)}
                                            className="w-8 h-8 flex items-center justify-center text-gray-400 hover:text-[#00ff88] transition-colors"
                                        >
                                            <Plus size={14} strokeWidth={3} />
                                        </button>
                                    </div>

                                    <button
                                        onClick={() => removeItem(item.id)}
                                        className="w-8 h-8 rounded-full bg-red-500/10 text-red-500 border border-red-500/10 flex items-center justify-center hover:bg-red-500 hover:text-white transition-all"
                                    >
                                        <X size={14} strokeWidth={3} />
                                    </button>
                                </div>
                            )}
                        </div>
                    ))}
                </div>

                {!readOnly && (
                    <div className="flex gap-2 mt-4">
                        <button
                            onClick={() => navigate("/inventory")}
                            className="flex-1 py-3 bg-[#00ff88]/10 text-[#00ff88] rounded-xl text-[10px] font-black uppercase tracking-widest hover:bg-[#00ff88]/20 transition-all border border-[#00ff88]/20"
                        >
                            查看詳情
                        </button>
                        <button
                            onClick={clearTempDetections}
                            className="px-4 py-3 bg-white/5 text-gray-500 rounded-xl text-[10px] font-black uppercase tracking-widest hover:text-white transition-all border border-white/5"
                        >
                            清空
                        </button>
                    </div>
                )}
            </div>
        </div>
    );
}
