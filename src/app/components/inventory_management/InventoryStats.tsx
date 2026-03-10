import { Package, AlertCircle } from "lucide-react";

interface InventoryStatsProps {
    totalItems: number;
    freshItems: number;
}

export function InventoryStats({ totalItems, freshItems }: InventoryStatsProps) {
    return (
        <div className="grid grid-cols-2 gap-4 px-6 py-4">
            <div className="bg-[#1a4d3d]/40 backdrop-blur-md border border-white/10 rounded-[2rem] p-6 shadow-2xl relative group overflow-hidden">
                <div className="absolute -top-4 -right-4 text-[#00ff88]/5 group-hover:text-[#00ff88]/10 transition-colors">
                    <Package size={80} strokeWidth={1} />
                </div>
                <div className="text-3xl font-black text-white mb-1">{totalItems}</div>
                <div className="text-[10px] font-black text-gray-400 uppercase tracking-widest opacity-60">Total Units</div>
            </div>

            <div className="bg-[#1a4d3d]/40 backdrop-blur-md border border-white/10 rounded-[2rem] p-6 shadow-2xl relative overflow-hidden group">
                <div className="absolute top-5 right-5 w-2 h-2 rounded-full bg-[#00ff88] shadow-[0_0_12px_#00ff88] animate-pulse" />
                <div className="text-3xl font-black text-[#00ff88] mb-1">{freshItems}</div>
                <div className="text-[10px] font-black text-gray-400 uppercase tracking-widest opacity-60">Verified SKU</div>
            </div>
        </div>
    );
}
