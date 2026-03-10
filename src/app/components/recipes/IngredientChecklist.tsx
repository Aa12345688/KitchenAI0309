import { CheckCircle2 } from "lucide-react";

interface IngredientChecklistProps {
    ingredients: string[];
    checkedItems: boolean[];
    onToggle: (index: number) => void;
    progress: number;
}

export function IngredientChecklist({
    ingredients,
    checkedItems,
    onToggle,
    progress
}: IngredientChecklistProps) {
    return (
        <div className="mb-10 bg-[#1a4d3d]/20 rounded-[2.5rem] p-8 border border-white/5">
            <h3 className="text-[10px] font-black text-gray-400 uppercase tracking-[0.2em] mb-6 flex items-center gap-2">
                <div className="w-1.5 h-1.5 rounded-full bg-[#00ff88] shadow-[0_0_8px_#00ff88]" />
                食材確認
            </h3>

            <div className="space-y-4">
                {ingredients.map((ingredient, index) => (
                    <button
                        key={index}
                        onClick={() => onToggle(index)}
                        className="w-full flex items-center gap-4 group text-left"
                    >
                        <div className="flex-shrink-0">
                            {checkedItems[index] ? (
                                <div className="w-6 h-6 rounded-lg bg-[#00ff88] flex items-center justify-center shadow-[0_0_10px_rgba(0,255,136,0.3)]">
                                    <CheckCircle2 size={16} className="text-[#0f2e24]" strokeWidth={3} />
                                </div>
                            ) : (
                                <div className="w-6 h-6 rounded-lg border-2 border-white/10 group-hover:border-[#00ff88]/50 transition-colors" />
                            )}
                        </div>
                        <span className={`text-[11px] font-bold uppercase tracking-tight transition-all ${checkedItems[index] ? 'text-white/20 line-through' : 'text-white'}`}>
                            {ingredient}
                        </span>
                    </button>
                ))}
            </div>

            <div className="mt-8 pt-6 border-t border-white/5">
                <div className="flex justify-between items-end mb-3 px-1">
                    <span className="text-[9px] font-black text-white/30 uppercase tracking-widest">備料進度</span>
                    <span className="text-xs font-black text-[#00ff88]">{progress}%</span>
                </div>
                <div className="w-full h-2 bg-white/5 rounded-full overflow-hidden p-0.5 border border-white/5">
                    <div
                        className="h-full bg-[#00ff88] rounded-full transition-all duration-500 shadow-[0_0_10px_#00ff88]"
                        style={{ width: `${progress}%` }}
                    />
                </div>
            </div>
        </div>
    );
}
