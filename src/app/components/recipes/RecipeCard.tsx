import { Clock, TrendingUp, ChevronRight } from "lucide-react";

interface RecipeCardProps {
    recipe: any;
    onClick: () => void;
    getCategoryLabel: (cat: string) => string;
}

export function RecipeCard({ recipe, onClick, getCategoryLabel }: RecipeCardProps) {
    return (
        <div
            onClick={onClick}
            className="group relative h-[400px] rounded-[3rem] overflow-hidden bg-gray-900 shadow-2xl transition-all duration-500 cursor-pointer"
        >
            <img
                src={recipe.image}
                alt={recipe.name}
                className="absolute inset-0 w-full h-full object-cover group-hover:scale-110 transition-transform duration-700 opacity-60"
            />
            <div className="absolute inset-0 bg-gradient-to-t from-[#0f2e24] via-[#0f2e24]/30 to-transparent" />

            <div className="absolute top-6 left-6 right-6 flex justify-between items-start">
                <div className="flex flex-col gap-2">
                    <div className="bg-[#00ff88] text-[#0f2e24] text-[9px] font-black px-4 py-1.5 rounded-full shadow-lg uppercase tracking-widest">
                        減少食物浪費
                    </div>
                    <div className="bg-white/10 backdrop-blur-md text-white border border-white/10 text-[8px] font-black px-3 py-1 rounded-full uppercase tracking-tighter">
                        {getCategoryLabel(recipe.category)}
                    </div>
                </div>

                <div className="bg-[#0f2e24]/80 backdrop-blur-md rounded-2xl p-2 border border-[#00ff88]/30 flex flex-col items-center min-w-[50px] shadow-2xl">
                    <span className="text-[12px] font-black text-[#00ff88]">{recipe.matchScore}%</span>
                    <span className="text-[7px] font-black text-white/40 uppercase tracking-tighter">匹配度</span>
                </div>
            </div>

            <div className="absolute bottom-6 left-6 right-6 p-2">
                <div className="flex items-center gap-2 mb-2">
                    <div className="w-1.5 h-1.5 rounded-full bg-[#00ff88] animate-pulse" />
                    <span className="text-[9px] font-black text-[#00ff88]/80 uppercase tracking-widest">準備執行</span>
                </div>

                <h3 className="font-black text-xl text-white mb-3 uppercase tracking-tight group-hover:text-[#00ff88] transition-colors leading-tight">{recipe.name}</h3>

                <div className="flex items-center gap-6 mb-6 text-[10px] font-black text-white/50 uppercase tracking-widest">
                    <div className="flex items-center gap-2">
                        <Clock size={14} className="text-[#00ff88]" strokeWidth={3} />
                        <span>{recipe.time}</span>
                    </div>
                    <div className="flex items-center gap-2">
                        <TrendingUp size={14} className="text-[#00ff88]" strokeWidth={3} />
                        <span>{recipe.difficulty}</span>
                    </div>
                </div>

                <button
                    className="w-full bg-[#00ff88] text-[#0f2e24] py-4 rounded-2xl font-black text-xs uppercase tracking-widest shadow-xl active:scale-95 transition-all flex items-center justify-center gap-3"
                >
                    <span>開始烹飪</span>
                    <ChevronRight size={16} strokeWidth={3} />
                </button>
            </div>
        </div>
    );
}
