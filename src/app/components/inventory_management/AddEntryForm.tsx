import { X } from "lucide-react";
import { useState } from "react";

interface AddEntryFormProps {
    onAdd: (item: { name: string; quantity: number; category: string }) => void;
    onDismiss: () => void;
    categories: string[];
}

export function AddEntryForm({ onAdd, onDismiss, categories }: AddEntryFormProps) {
    const [name, setName] = useState("");
    const [quantity, setQuantity] = useState(1);
    const [category, setCategory] = useState("Other");

    const handleSubmit = () => {
        if (name.trim()) {
            onAdd({ name: name.trim(), quantity, category });
            setName("");
            setQuantity(1);
            onDismiss();
        }
    };

    return (
        <div className="px-6 py-4 animate-in slide-in-from-top-4 duration-300">
            <div className="bg-[#1a4d3d] rounded-[2.5rem] p-8 border-2 border-[#00ff88]/20 shadow-2xl relative overflow-hidden">
                <div className="absolute top-0 right-0 p-4">
                    <button onClick={onDismiss} className="text-white/20 hover:text-white/50"><X size={20} /></button>
                </div>
                <div className="flex items-center gap-3 mb-6">
                    <h3 className="font-black text-xs tracking-widest uppercase text-white/50">Add New Entry</h3>
                </div>

                <div className="space-y-4">
                    <div>
                        <label className="text-[10px] font-black text-gray-500 uppercase tracking-widest mb-2 block">Item Name</label>
                        <input
                            type="text"
                            placeholder="e.g. Avocado"
                            value={name}
                            onChange={(e) => setName(e.target.value)}
                            className="w-full px-5 py-4 bg-[#0f2e24] rounded-2xl border border-white/10 focus:outline-none focus:border-[#00ff88] transition-all text-sm font-bold shadow-inner"
                        />
                    </div>

                    <div className="flex gap-4">
                        <div className="flex-1">
                            <label className="text-[10px] font-black text-gray-500 uppercase tracking-widest mb-2 block">Quantity</label>
                            <input
                                type="number"
                                min="1"
                                value={quantity}
                                onChange={(e) => setQuantity(parseInt(e.target.value) || 1)}
                                className="w-full px-5 py-4 bg-[#0f2e24] rounded-2xl border border-white/10 focus:outline-none focus:border-[#00ff88] transition-all text-sm font-bold shadow-inner"
                            />
                        </div>
                        <div className="flex-1">
                            <label className="text-[10px] font-black text-gray-500 uppercase tracking-widest mb-2 block">Category</label>
                            <select
                                value={category}
                                onChange={(e) => setCategory(e.target.value)}
                                className="w-full px-5 py-4 bg-[#0f2e24] rounded-2xl border border-white/10 focus:outline-none focus:border-[#00ff88] transition-all text-sm font-bold appearance-none shadow-inner"
                            >
                                {categories.filter(c => c !== "All").map(c => (
                                    <option key={c} value={c}>{c}</option>
                                ))}
                            </select>
                        </div>
                    </div>

                    <div className="pt-2">
                        <button
                            onClick={handleSubmit}
                            className="w-full py-4 bg-[#00ff88] text-[#0f2e24] rounded-2xl font-black uppercase tracking-widest text-xs shadow-[0_10px_20px_rgba(0,255,136,0.3)] active:scale-95 transition-all"
                        >
                            Confirm Registry
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
}
