import { useNavigate, useParams } from "react-router";
import { useState, useEffect } from "react";
import {
    Camera, Sparkles, X, Plus, Minus, Package,
    Trash2, Search, Share2, ChefHat,
    User, Settings, HelpCircle, LogOut, ChevronRight,
    BookOpen, Clock, Users, Loader2, Mic, Edit2, AlertTriangle, Snowflake, Moon
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { PageHeader } from "../components/Shared";
import { useIngredients } from "../services/IngredientContext";
import { formatRelativeTime } from "../services/timeUtils";
import { useCamera } from "../hooks/useCamera";
import { CameraView } from "../components/scanner/CameraView";

// Use consolidated components from inventory_management
import { DetectionRow } from "../components/inventory_management/DetectionRow";
import { DetectionSummary } from "../components/inventory_management/DetectionSummary";
import { InventoryStats } from "../components/inventory_management/InventoryStats";
import { AddEntryForm } from "../components/inventory_management/AddEntryForm";

import { getRecommendedRecipes, recipeDatabase } from "../data/recipes";
import { llmService } from "../services/llmService";
import { RecipeCard } from "../components/recipes/RecipeCard";
import { IngredientCloud } from "../components/recipes/IngredientCloud";
import { RecipeHero } from "../components/recipes/RecipeHero";
import { IngredientChecklist } from "../components/recipes/IngredientChecklist";
import { CookingProtocol } from "../components/recipes/CookingProtocol";

// --- Scanner Page ---
export function Scanner() {
    const navigate = useNavigate();
    const { scannedItems } = useIngredients();
    const { videoRef } = useCamera();

    return (
        <div className="pb-24">
            <div className="flex flex-col items-center justify-center px-6 pt-12 pb-4">
                <CameraView videoRef={videoRef} />

                <p className="text-center text-gray-400 text-xs mt-8 px-10 leading-relaxed uppercase tracking-widest font-medium opacity-60">將鏡頭對準食材<br />AI 將自動辨識並同步庫存</p>
            </div>
        </div>
    );
}

// --- Ingredients Page ---
export function Ingredients() {
    const navigate = useNavigate();
    const { scannedItems, updateQuantity, removeItem, clearAll } = useIngredients();
    return (
        <div className="pb-32">
            <PageHeader showBackButton title="最近辨識" rightAction={<button onClick={clearAll} className="p-2 sm:p-2.5 bg-red-500/10 hover:bg-red-500 hover:text-white rounded-2xl border border-red-500/10 text-red-500"><Trash2 size={20} className="stroke-[2.5]" /></button>} />
            <div className="px-6 py-6">
                <h2 className="text-[11px] font-black text-white/30 uppercase tracking-[0.2em] mb-6 px-1 flex items-center gap-2"><div className="w-1 h-1 rounded-full bg-[#00ff88]" />掃描紀錄</h2>
                {scannedItems.length === 0 ? (
                    <div className="flex flex-col items-center justify-center py-24 text-center bg-white/5 rounded-[3.5rem] border border-white/5">
                        <div className="relative mb-8 group"><div className="absolute inset-0 bg-[#00ff88]/5 rounded-full blur-3xl" /><div className="relative w-24 h-24 bg-[#1a4d3d]/50 rounded-[2rem] border border-white/10 flex items-center justify-center shadow-2xl"><Plus size={40} className="text-[#00ff88]/20" /></div></div>
                        <h3 className="text-sm font-black text-white/30 uppercase tracking-widest mb-4">目前無數據暫存</h3>
                        <button onClick={() => navigate("/")} className="flex items-center gap-3 bg-[#00ff88] text-[#0f2e24] px-8 py-4 rounded-2xl font-black uppercase text-[10px] shadow-xl">啟動感測器</button>
                    </div>
                ) : (
                    <div className="space-y-3">{scannedItems.slice(0, 10).map((item) => (<DetectionRow key={item.id} item={item} onUpdate={updateQuantity} onRemove={removeItem} />))}</div>
                )}
            </div>
        </div>
    );
}

// --- Recipes Page ---
export function Recipes() {
    const navigate = useNavigate();
    const { scannedItems, recommendedRecipes, setRecipes } = useIngredients();
    const [isLoading, setIsLoading] = useState(true);

    useEffect(() => {
        if (scannedItems.length > 0 && recommendedRecipes.length === 0) {
            const fetchRecipes = async () => {
                setIsLoading(true);
                try {
                    const res = await llmService.generateRecipes({ ingredients: scannedItems.map(i => i.name) });
                    setRecipes(res);
                } catch (error) {
                    setRecipes(getRecommendedRecipes(scannedItems)); // Local fallback
                } finally { setIsLoading(false); }
            };
            fetchRecipes();
        } else { setIsLoading(false); }
    }, [scannedItems, recommendedRecipes, setRecipes]);

    return (
        <div className="pb-24">
            <PageHeader showBackButton title="AI 推薦食譜" rightAction={<button className="p-2.5 bg-white/5 rounded-2xl hover:bg-white/10"><Share2 size={20} className="text-white" /></button>} />
            <div className="px-6 py-4">
                <IngredientCloud items={scannedItems} onAddMore={() => navigate("/inventory")} />
                {isLoading ? (
                    <div className="flex flex-col items-center justify-center py-20 gap-6 bg-white/5 rounded-[3rem] border border-white/10"><div className="relative w-16 h-16"><div className="absolute inset-0 border-2 border-[#00ff88]/20 rounded-full" /><div className="absolute inset-0 border-2 border-[#00ff88] border-t-transparent rounded-full animate-spin" /><Sparkles className="absolute inset-0 m-auto text-[#00ff88] animate-pulse" size={20} /></div><div className="text-center"><h3 className="text-[#00ff88] font-black text-xs uppercase animate-pulse mb-1">運算中...</h3><p className="text-gray-500 text-[9px] font-bold uppercase">正在分析口味分佈</p></div></div>
                ) : recommendedRecipes.length > 0 ? (
                    <div className="space-y-8"><div className="bg-[#1a4d3d]/40 backdrop-blur-md rounded-[2.5rem] p-6 border border-[#00ff88]/20 flex items-start gap-4"><div className="w-12 h-12 rounded-2xl bg-[#00ff88] flex items-center justify-center flex-shrink-0 shadow-lg"><ChefHat size={24} className="text-[#0f2e24]" strokeWidth={2.5} /></div><div><h3 className="font-black text-xs text-white uppercase mb-1">AI 神經網路推薦</h3><p className="text-[10px] text-gray-400 font-bold uppercase leading-tight">已優化 <span className="text-[#00ff88]">{recommendedRecipes.length} 個相容節點</span> <br />惜食減廢協議已啟動</p></div></div><div className="grid grid-cols-1 gap-10">{recommendedRecipes.map((r) => (<RecipeCard key={r.id} recipe={r} onClick={() => navigate(`/recipe/${r.id}`)} getCategoryLabel={(c) => c === "vegetable" ? "蔬菜" : c === "fruit" ? "水果" : c === "meat" ? "肉類" : "綜合"} />))}</div></div>
                ) : (
                    <div className="text-center py-24 px-8 bg-white/5 rounded-[3rem] border-2 border-dashed border-white/5"><div className="w-20 h-20 mx-auto mb-6 bg-[#00ff88]/5 rounded-full flex items-center justify-center"><ChefHat size={40} className="text-[#00ff88]/20" /></div><h4 className="text-white font-black text-sm uppercase mb-2">未發現相容方案</h4><button onClick={() => navigate("/")} className="inline-flex items-center gap-3 px-8 py-4 bg-[#00ff88] text-[#0f2e24] rounded-2xl font-black uppercase text-[10px]">返回掃描</button></div>
                )}
            </div>
        </div>
    );
}
// --- Edit Item Modal ---
function EditItemModal({ item, onSave, onDismiss }: { item: any, onSave: (id: string, updates: any) => void, onDismiss: () => void }) {
    const [name, setName] = useState(item.name);
    const [category, setCategory] = useState(item.category || "其他");
    const [storageType, setStorageType] = useState(item.storageType || "fridge");
    const [expiryDays, setExpiryDays] = useState(item.expiryDays || 7);

    return (
        <div className="fixed inset-0 z-[100] bg-[#0f2e24]/90 backdrop-blur-xl flex items-center justify-center p-6">
            <div className="bg-[#1a4d3d] w-full max-w-sm rounded-[2.5rem] p-6 border border-white/10 shadow-2xl relative">
                <button onClick={onDismiss} className="absolute right-6 top-6 w-8 h-8 rounded-full bg-white/5 flex items-center justify-center text-gray-500 hover:text-white">
                    <X size={16} />
                </button>
                <div className="w-12 h-12 bg-[#00ff88]/10 rounded-2xl flex items-center justify-center mb-6">
                    <Edit2 size={24} className="text-[#00ff88]" />
                </div>
                <h3 className="text-xl font-black text-white uppercase tracking-wider mb-6">編輯食材資訊</h3>

                <div className="space-y-4 mb-8">
                    <div>
                        <label className="text-[10px] font-black uppercase text-gray-500 mb-2 block tracking-widest">食材名稱</label>
                        <input type="text" value={name} onChange={(e) => setName(e.target.value)} className="w-full bg-[#0f2e24] border border-white/10 rounded-xl px-4 py-3 text-white font-bold focus:border-[#00ff88] outline-none transition-colors" />
                    </div>
                    <div>
                        <label className="text-[10px] font-black uppercase text-gray-500 mb-2 block tracking-widest">分類標籤</label>
                        <div className="flex flex-wrap gap-2">
                            {["蔬菜", "水果", "肉類", "乳製品", "五穀", "其他"].map(c => (
                                <button key={c} onClick={() => setCategory(c)} className={`px-4 py-2 rounded-lg text-[10px] font-black uppercase border transition-all ${category === c ? 'bg-[#00ff88] text-[#0f2e24] border-[#00ff88]' : 'bg-white/5 text-gray-400 border-white/10'}`}>{c}</button>
                            ))}
                        </div>
                    </div>
                    <div className="grid grid-cols-2 gap-4">
                        <div>
                            <label className="text-[10px] font-black uppercase text-gray-500 mb-2 block tracking-widest">保存位置</label>
                            <div className="flex bg-[#0f2e24] p-1 rounded-xl border border-white/10">
                                <button onClick={() => setStorageType('fridge')} className={`flex-1 py-2 rounded-lg text-[10px] font-black uppercase transition-all flex items-center justify-center gap-1 ${storageType === 'fridge' ? 'bg-white/10 text-white' : 'text-gray-500'}`}><ChefHat size={12} />冷藏</button>
                                <button onClick={() => setStorageType('freezer')} className={`flex-1 py-2 rounded-lg text-[10px] font-black uppercase transition-all flex items-center justify-center gap-1 ${storageType === 'freezer' ? 'bg-blue-500/20 text-blue-300' : 'text-gray-500'}`}><Snowflake size={12} />冷凍</button>
                            </div>
                        </div>
                        <div>
                            <label className="text-[10px] font-black uppercase text-gray-500 mb-2 block tracking-widest">保鮮期 (天)</label>
                            <input type="number" value={expiryDays} onChange={(e) => setExpiryDays(parseInt(e.target.value) || 0)} className="w-full bg-[#0f2e24] border border-white/10 rounded-xl px-4 py-3 text-white font-bold focus:border-[#00ff88] outline-none transition-colors text-center" />
                        </div>
                    </div>
                </div>

                <button onClick={() => onSave(item.id, { name, category, storageType, expiryDays })} className="w-full bg-[#00ff88] text-[#0f2e24] py-4 rounded-xl font-black text-sm uppercase tracking-widest shadow-lg active:scale-95 transition-all">
                    更新資料 (Update)
                </button>
            </div>
        </div>
    );
}

// --- 庫存管理主要頁面 (Inventory Management Page) ---
export function Inventory() {
    const navigate = useNavigate();
    // 從 Context 獲取全域狀態與操作方法
    const { scannedItems, addItem, updateQuantity, removeIngredient, selectedIds, toggleSelection, generateRecipe, wasteHistory, updateItem } = useIngredients();
    const [isGenerating, setIsGenerating] = useState(false);

    // --- 手動調整參數：分頁、搜尋與過濾狀態 ---
    const [search, setSearch] = useState(""); // 搜尋關鍵字
    const [cat, setCat] = useState("全部"); // 食材分類標籤過濾
    const [showForm, setShowForm] = useState(false); // 是否顯示手動輸入表單
    const [storageTab, setStorageTab] = useState<"fridge" | "freezer">("fridge"); // 切換 冷藏/冷凍 分頁
    const [editingItem, setEditingItem] = useState<any>(null); // 當前正在編輯的食材對象
    const [isListening, setIsListening] = useState(false); // 語音辨識狀態 (啟動/停止)

    const startVoiceInput = () => {
        const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
        if (!SpeechRecognition) return alert("您的瀏覽器不支援語音辨識功能！");

        const recognition = new SpeechRecognition();
        recognition.lang = 'zh-TW';
        recognition.start();
        setIsListening(true);

        recognition.onresult = (event: any) => {
            const transcript = event.results[0][0].transcript;
            setSearch(transcript);
            setIsListening(false);
        };
        recognition.onerror = () => setIsListening(false);
        recognition.onend = () => setIsListening(false);
    };

    const filtered = scannedItems.filter(i =>
        (i.storageType || "fridge") === storageTab &&
        (cat === "全部" || i.category === cat) &&
        i.name.toLowerCase().includes(search.toLowerCase())
    );
    const total = scannedItems.reduce((s, i) => s + i.quantity, 0);

    const handleSaveEdit = (id: string, updates: any) => {
        updateItem(id, updates);
        setEditingItem(null);
    };

    return (
        <div className="pb-24">
            <PageHeader showBackButton title="食材清單" rightAction={<button onClick={() => setShowForm(!showForm)} className="p-1.5 bg-[#00ff88] rounded-xl shadow-lg"><Plus size={20} className="text-[#0f2e24] stroke-[3]" /></button>} />

            <div className="bg-[#0f2e24] sticky top-[64px] z-20 pb-4 px-6 py-4">
                <div className="flex bg-white/5 p-1 rounded-2xl mb-4">
                    <button onClick={() => setStorageTab('fridge')} className={`flex-1 py-2.5 rounded-xl text-xs font-black uppercase transition-all flex items-center justify-center gap-2 ${storageTab === 'fridge' ? 'bg-[#00ff88] text-[#0f2e24]' : 'text-gray-400'}`}><ChefHat size={16} />冷藏庫</button>
                    <button onClick={() => setStorageTab('freezer')} className={`flex-1 py-2.5 rounded-xl text-xs font-black uppercase transition-all flex items-center justify-center gap-2 ${storageTab === 'freezer' ? 'bg-blue-400 text-[#0f2e24]' : 'text-gray-400'}`}><Snowflake size={16} />冷凍庫</button>
                </div>

                <div className="relative flex items-center gap-2">
                    <div className="relative flex-1">
                        <Search className="absolute left-4 top-1/2 -translate-y-1/2 text-gray-500" size={18} />
                        <input type="text" placeholder="搜尋食材..." value={search} onChange={(e) => setSearch(e.target.value)} className="w-full pl-12 pr-4 py-4 bg-white/5 border border-white/10 rounded-2xl focus:border-[#00ff88] text-sm font-bold placeholder:text-gray-600 outline-none" />
                    </div>
                    <button onClick={startVoiceInput} className={`p-4 rounded-2xl transition-all ${isListening ? 'bg-red-500 text-white animate-pulse' : 'bg-white/5 text-gray-400 border border-white/10'}`}>
                        <Mic size={20} />
                    </button>
                </div>

                <div className="flex gap-2 overflow-x-auto no-scrollbar mt-4">
                    {["全部", "蔬菜", "水果", "乳製品", "肉類", "五穀", "其他"].map(c => (
                        <button key={c} onClick={() => setCat(c)} className={`px-5 py-2 rounded-xl text-[10px] font-black uppercase whitespace-nowrap border transition-colors ${cat === c ? (storageTab === 'fridge' ? "bg-[#00ff88] text-[#0f2e24] border-[#00ff88]" : "bg-blue-400 text-[#0f2e24] border-blue-400") : "bg-white/5 text-gray-500 border-white/10"}`}>{c}</button>
                    ))}
                </div>
            </div>

            <div className="px-6 mb-8 mt-2">
                <WasteChart data={wasteHistory} />
            </div>

            <div className="px-6 mb-2">
                <button
                    onClick={async () => {
                        setIsGenerating(true);
                        try {
                            await generateRecipe();
                            navigate("/recipes");
                        } catch (e: any) {
                            alert(e.message);
                        } finally {
                            setIsGenerating(false);
                        }
                    }}
                    disabled={isGenerating || selectedIds.length === 0}
                    className={`w-full ${storageTab === 'fridge' ? 'bg-[#00ff88]' : 'bg-blue-400'} text-[#0f2e24] py-5 rounded-[2rem] font-black text-xs uppercase tracking-widest shadow-[0_10px_30px_rgba(0,255,136,0.2)] flex items-center justify-center gap-3 disabled:opacity-50 transition-all hover:scale-[1.02] active:scale-[0.98]`}
                >
                    {isGenerating ? <Loader2 size={18} className="animate-spin" /> : <Sparkles size={18} />}
                    {isGenerating ? "Synthesizing..." : "生成 AI 食譜方案"}
                </button>
            </div>

            <InventoryStats totalItems={total} freshItems={scannedItems.length} />

            {showForm && (<AddEntryForm onAdd={addItem} onDismiss={() => setShowForm(false)} categories={["全部", "蔬菜", "水果", "乳製品", "肉類", "五穀", "其他"]} />)}

            <div className="px-6 py-4">
                <h3 className="font-black text-xs uppercase text-white/30 mb-4 px-2">存貨紀錄 ({filtered.length})</h3>
                {filtered.length === 0 ? (
                    <div className="text-center py-20 bg-white/5 rounded-[3rem] border border-white/5">
                        <Package size={48} className="mx-auto mb-4 text-white/10" />
                        <p className="text-[10px] font-bold text-gray-500">該庫存區域為空</p>
                    </div>
                ) : (
                    <div className="space-y-3">
                        {filtered.map(i => {
                            const daysPassed = Math.floor((Date.now() - (i.timestamp || Date.now())) / (1000 * 60 * 60 * 24));
                            const expiryDays = i.expiryDays || 7;
                            const daysLeft = expiryDays - daysPassed;
                            const isExpired = daysLeft < 0;
                            const isWarning = daysLeft >= 0 && daysLeft <= 2;

                            return (
                                <div key={i.id} className={`bg-[#1a4d3d]/30 rounded-2xl p-4 border ${i.isSpoiled || isExpired ? 'border-red-500/50 shadow-[0_0_15px_rgba(239,68,68,0.1)]' : isWarning ? 'border-amber-400/50 shadow-[0_0_15px_rgba(251,191,36,0.1)]' : 'border-white/5'} flex flex-col gap-3 transition-all relative overflow-hidden group`}>

                                    <div className="flex items-center gap-4">
                                        <button
                                            onClick={() => toggleSelection(i.id)}
                                            disabled={i.isSpoiled || isExpired}
                                            className={`w-6 h-6 rounded-lg border-2 flex items-center justify-center flex-shrink-0 transition-all ${i.isSpoiled || isExpired ? 'opacity-20 cursor-not-allowed border-gray-600' : selectedIds.includes(i.id) ? (storageTab === 'fridge' ? 'bg-[#00ff88] border-[#00ff88]' : 'bg-blue-400 border-blue-400') : 'bg-transparent border-white/20'}`}
                                        >
                                            {selectedIds.includes(i.id) && !i.isSpoiled && !isExpired && <div className="w-3 h-3 bg-[#0f2e24] rounded-sm" />}
                                            {(i.isSpoiled || isExpired) && <X size={12} className="text-red-500" />}
                                        </button>

                                        <div className="w-12 h-12 rounded-xl bg-[#0f2e24] flex items-center justify-center flex-shrink-0 relative">
                                            <Package size={20} className={i.isSpoiled || isExpired ? "text-red-500" : storageTab === 'fridge' ? "text-[#00ff88]" : "text-blue-400"} />
                                            {(i.isSpoiled || isExpired) && <div className="absolute inset-0 bg-red-500/10 rounded-xl" />}
                                        </div>

                                        <div className="flex-1 min-w-0">
                                            <h4 className={`font-black text-sm truncate uppercase ${i.isSpoiled || isExpired ? 'text-red-500 line-through' : 'text-white'}`}>{i.name}</h4>
                                            <div className="flex flex-wrap items-center gap-2 mt-1">
                                                <span className={`text-[8px] font-black uppercase px-2 py-0.5 rounded-md ${i.isSpoiled ? 'bg-red-500/10 text-red-500' : storageTab === 'fridge' ? 'bg-[#00ff88]/10 text-[#00ff88]' : 'bg-blue-400/10 text-blue-400'}`}>
                                                    {i.isSpoiled ? "品質異常" : (i.category || "其他")}
                                                </span>
                                                {!i.isSpoiled && (
                                                    <span className={`text-[8px] font-black uppercase px-2 py-0.5 rounded-md flex items-center gap-1 ${isExpired ? 'bg-red-500/20 text-red-400' : isWarning ? 'bg-amber-400/20 text-amber-400' : 'bg-white/5 text-gray-400'}`}>
                                                        <Clock size={8} />
                                                        {isExpired ? `已過期 ${Math.abs(daysLeft)} 天` : isWarning ? `剩餘 ${daysLeft} 天` : `保鮮 ${daysLeft} 天`}
                                                    </span>
                                                )}
                                                <span className="text-[8px] font-black uppercase px-2 py-0.5 rounded-md flex items-center gap-1 bg-white/5 text-gray-400">
                                                    {new Date(i.timestamp || Date.now()).toLocaleString('zh-TW', { month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit' })}
                                                </span>
                                            </div>
                                        </div>

                                        <div className="flex items-center gap-2">
                                            <button onClick={() => setEditingItem(i)} className="w-8 h-8 rounded-full bg-white/5 text-gray-400 flex items-center justify-center hover:text-white hover:bg-white/10 transition-colors">
                                                <Edit2 size={14} />
                                            </button>
                                        </div>
                                    </div>

                                    <div className="flex items-center justify-end gap-2 border-t border-white/5 pt-3">
                                        <div className="flex items-center bg-[#0f2e24]/80 rounded-full p-1 border border-white/10">
                                            <button onClick={() => updateQuantity(i.id, -1)} className={`w-7 h-7 flex items-center justify-center text-gray-500 hover:${storageTab === 'fridge' ? 'text-[#00ff88]' : 'text-blue-400'}`}><Minus size={12} strokeWidth={3} /></button>
                                            <span className="w-8 text-center font-black text-white text-xs">{i.quantity}</span>
                                            <button onClick={() => updateQuantity(i.id, 1)} className={`w-7 h-7 flex items-center justify-center text-gray-500 hover:${storageTab === 'fridge' ? 'text-[#00ff88]' : 'text-blue-400'}`}><Plus size={12} strokeWidth={3} /></button>
                                        </div>
                                        <button onClick={() => removeIngredient(i.id)} className="w-8 h-8 rounded-full bg-red-500/10 text-red-500 flex items-center justify-center transition-all hover:bg-red-500 hover:text-white"><Trash2 size={12} strokeWidth={3} /></button>
                                    </div>

                                    {isWarning && !isExpired && (
                                        <div className="absolute top-0 right-0 w-8 h-8 overflow-hidden">
                                            <div className="absolute top-[-10px] right-[-10px] bg-amber-400 w-12 h-12 rotate-45 transform origin-bottom-left flex items-end justify-center pb-1">
                                                <AlertTriangle size={8} className="text-[#0f2e24] -rotate-45" strokeWidth={3} />
                                            </div>
                                        </div>
                                    )}
                                </div>
                            );
                        })}
                    </div>
                )}
            </div>

            {editingItem && (
                <EditItemModal
                    item={editingItem}
                    onSave={handleSaveEdit}
                    onDismiss={() => setEditingItem(null)}
                />
            )}
        </div>
    );
}

// --- Waste Chart Component ---
function WasteChart({ data }: { data: any[] }) {
    const [period, setPeriod] = useState("week");
    const [selectedMonth, setSelectedMonth] = useState(new Date().getMonth()); // 0-11

    const months = ["一月", "二月", "三月", "四月", "五月", "六月", "七月", "八月", "九月", "十月", "十一月", "十二月"];

    // 生成假資料的方法 (真實環境應從後端或 DB 讀取)
    // 預設清空，這裡模擬根據選擇的月份產生不同的全空數據
    const generateMonthlyData = (monthIndex: number) => {
        // 在這裡可以放入特定的假數據，或回傳空。用戶說「清空」所以預設給 0
        return [
            { date: "第1週", amount: 0 },
            { date: "第2週", amount: 0 },
            { date: "第3週", amount: 0 },
            { date: "第4週", amount: 0 }
        ];
    };

    const generateWeeklyData = () => {
        return [
            { date: "2024-03-04", amount: 0 },
            { date: "2024-03-05", amount: 0 },
            { date: "2024-03-06", amount: 0 },
            { date: "2024-03-07", amount: 0 },
            { date: "2024-03-08", amount: 0 },
            { date: "2024-03-09", amount: 0 },
            { date: "2024-03-10", amount: 0 }
        ];
    }

    // 根據當前選擇渲染數據
    const displayData = period === "week" ? generateWeeklyData() : generateMonthlyData(selectedMonth);
    const max = Math.max(...displayData.map(d => d.amount), 5); // Default to smaller max since units are portions/items
    const chartHeight = 100;

    return (
        <div className="bg-[#1a4d3d]/30 rounded-[2.5rem] p-6 border border-white/5 mb-8">
            <div className="flex flex-col gap-4 mb-6">
                <div className="flex items-center justify-between">
                    <div>
                        <h3 className="text-[10px] font-black text-[#00ff88] uppercase tracking-widest mb-1">食物浪費統計</h3>
                        <p className="text-[8px] text-gray-400 font-bold uppercase">單位: 份數</p>
                    </div>
                    <div className="flex bg-[#0f2e24] p-1 rounded-xl border border-white/5">
                        {["week", "month"].map(p => (
                            <button
                                key={p}
                                onClick={() => setPeriod(p)}
                                className={`px-3 py-1.5 rounded-lg text-[8px] font-black uppercase transition-all ${period === p ? "bg-[#00ff88] text-[#0f2e24]" : "text-gray-500"}`}
                            >
                                {p === "week" ? "Week" : "Month"}
                            </button>
                        ))}
                    </div>
                </div>

                {/* 增加可以切換月份的選單 (僅在 Month 模式顯示) */}
                {period === "month" && (
                    <div className="flex items-center justify-between bg-black/20 p-2 rounded-xl border border-white/5">
                        <button
                            onClick={() => setSelectedMonth(prev => prev === 0 ? 11 : prev - 1)}
                            className="p-1 px-3 text-[#00ff88] bg-white/5 rounded-lg"
                        >{"<"}</button>
                        <span className="text-[10px] font-black text-white">{months[selectedMonth]}</span>
                        <button
                            onClick={() => setSelectedMonth(prev => prev === 11 ? 0 : prev + 1)}
                            className="p-1 px-3 text-[#00ff88] bg-white/5 rounded-lg"
                        >{">"}</button>
                    </div>
                )}
            </div>

            <div className="relative h-[120px] w-full flex items-end justify-between gap-2 px-2">
                {displayData.map((d, i) => {
                    const height = (d.amount / max) * chartHeight;
                    return (
                        <div key={i} className="flex-1 flex flex-col items-center gap-2 group">
                            <div className="relative w-full flex items-end justify-center">
                                <motion.div
                                    initial={{ height: 0 }}
                                    animate={{ height }}
                                    className={`w-full max-w-[12px] rounded-t-lg transition-all ${d.amount >= 3 ? 'bg-red-500 shadow-[0_0_10px_rgba(239,68,68,0.4)]' : 'bg-[#00ff88] shadow-[0_0_10px_rgba(0,255,136,0.2)]'}`}
                                />
                                <div className="absolute -top-6 opacity-0 group-hover:opacity-100 transition-opacity bg-black/80 px-2 py-1 rounded text-[8px] font-black text-white whitespace-nowrap z-10 border border-white/10">
                                    {d.amount} 份
                                </div>
                            </div>
                            <span className="text-[6px] font-black text-gray-500 uppercase flex flex-col items-center">
                                {period === "week" ? (
                                    <>
                                        <span>{d.date.split("-").slice(1).join("/")}</span>
                                    </>
                                ) : (
                                    <span>{d.date}</span>
                                )}
                            </span>
                        </div>
                    );
                })}
                <div className="absolute bottom-6 left-0 right-0 h-[1px] bg-white/5" />
            </div>
        </div>
    );
}

// --- Profile & Saved & Detail (Simplified Combined) ---
export function Profile() {
    const { settings, updateSettings } = useIngredients();

    return (
        <div className="pb-24 px-6 py-8"><PageHeader title="個人設定" />
            <div className="flex flex-col items-center mb-10">
                <div className="w-28 h-28 rounded-full bg-[#1a4d3d] border-4 border-[#00ff88]/20 flex items-center justify-center shadow-2xl mb-6">
                    <User size={48} className="text-[#00ff88]" strokeWidth={1} />
                </div>
                <h2 className="text-lg font-black text-white uppercase mb-2">AI 廚房大腦</h2>
            </div>

            <div className="bg-[#1a4d3d]/20 p-6 rounded-[2rem] border border-white/5 mb-8">
                <h3 className="text-[10px] font-black text-white/30 uppercase tracking-widest mb-6 px-1">系統設定</h3>
                <div className="space-y-4">
                    <div className="flex items-center justify-between p-4 bg-white/5 rounded-2xl border border-white/5">
                        <div className="flex items-center gap-4">
                            <div className="w-10 h-10 rounded-xl bg-[#00ff88]/10 flex items-center justify-center"><BookOpen size={18} className="text-[#00ff88]" /></div>
                            <div>
                                <div className="text-[10px] font-black text-white uppercase">通知</div>
                                <div className="text-[8px] font-bold text-gray-500 uppercase">接收過期與庫存示警</div>
                            </div>
                        </div>
                        <button
                            onClick={() => updateSettings({ notifications: !settings.notifications })}
                            className={`w-12 h-6 rounded-full relative transition-all duration-300 ${settings.notifications ? "bg-[#00ff88]" : "bg-white/10"}`}
                        >
                            <div className={`absolute top-1 w-4 h-4 rounded-full bg-white transition-all duration-300 ${settings.notifications ? "left-7" : "left-1"}`} />
                        </button>
                    </div>

                    <div className="flex items-center justify-between p-4 bg-white/5 rounded-2xl border border-white/5">
                        <div className="flex items-center gap-4">
                            <div className="w-10 h-10 rounded-xl bg-purple-500/10 flex items-center justify-center"><Moon size={18} className="text-purple-400" /></div>
                            <div>
                                <div className="text-[10px] font-black text-white uppercase">深色模式</div>
                                <div className="text-[8px] font-bold text-gray-500 uppercase">切換應用程式語意顏色</div>
                            </div>
                        </div>
                        <button
                            onClick={() => updateSettings({ darkMode: !settings.darkMode })}
                            className={`w-12 h-6 rounded-full relative transition-all duration-300 ${settings.darkMode ? "bg-purple-500" : "bg-white/10"}`}
                        >
                            <div className={`absolute top-1 w-4 h-4 rounded-full bg-white transition-all duration-300 ${settings.darkMode ? "left-7" : "left-1"}`} />
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
}

export function Saved() {
    const nav = useNavigate();
    return (
        <div className="pb-24 flex flex-col items-center justify-center px-8 py-24 text-center"><PageHeader showBackButton title="已暫存清單" /><div className="w-28 h-28 bg-white/5 rounded-[2.5rem] border border-white/10 flex items-center justify-center mb-10"><BookOpen size={48} className="text-[#00ff88]/20" /></div><h2 className="text-sm font-black text-white uppercase mb-4 opacity-50">暫無加入清單</h2><button onClick={() => nav("/")} className="bg-[#00ff88] text-[#0f2e24] px-10 py-5 rounded-2xl font-black uppercase text-[10px] shadow-lg">啟動掃描器</button></div>
    );
}

export function RecipeDetail() {
    const { id } = useParams();
    const nav = useNavigate();
    const { recommendedRecipes, scannedItems, setRecipes } = useIngredients();

    // 優先從全域推薦中找，找不到才去靜態庫 (Search context first, then static DB)
    const [recipe] = useState(() =>
        recommendedRecipes.find(r => r.id === id) ||
        recipeDatabase.find(r => r.id === id) ||
        {
            name: "AI 合成食譜",
            image: "https://images.unsplash.com/photo-1512621776951-a57141f2eefd?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&q=80&w=800",
            time: "15 分鐘",
            difficulty: "簡單",
            requiredIngredients: ["番茄", "菠菜"],
            optionalIngredients: [],
            description: "智慧生成食譜。"
        }
    );
    const [checked, setChecked] = useState<boolean[]>(new Array(recipe.requiredIngredients.length).fill(false));

    return (
        <div className="pb-32"><PageHeader showBackButton title="烹飪指南" /><RecipeHero image={recipe.image} name={recipe.name} />
            <div className="px-6 py-8"><div className="grid grid-cols-3 gap-3 mb-10">{[{ i: Clock, v: recipe.time }, { i: ChefHat, v: recipe.difficulty }, { i: Users, v: "2-3人份" }].map((s, i) => (<div key={i} className="bg-white/5 rounded-2xl p-4 text-center"><s.i className="w-4 h-4 mx-auto mb-2 text-[#00ff88]" /><div className="text-xs font-black text-white">{s.v}</div></div>))}</div>
                <IngredientChecklist ingredients={recipe.requiredIngredients} checkedItems={checked} onToggle={(i) => { const n = [...checked]; n[i] = !n[i]; setChecked(n); }} progress={Math.round((checked.filter(Boolean).length / recipe.requiredIngredients.length) * 100)} />
                <CookingProtocol steps={recipe.steps || [{ title: "初始化", description: "準備食材。" }, { title: "執行", description: "標準烹飪。" }]} />

                {/* 重新分析按鈕放置於最底部 */}
                <div className="mt-12 mb-8 px-2">
                    <button
                        onClick={async () => {
                            try {
                                const ingredientsToUse = scannedItems.map(i => i.name);
                                const res = await llmService.generateRecipes({ ingredients: ingredientsToUse });
                                setRecipes(res);
                                alert("AI 食譜已重新合成！");
                            } catch (e: any) {
                                alert("更新食譜失敗");
                            }
                        }}
                        className="w-full flex items-center justify-center gap-3 bg-white/5 border border-white/10 py-5 rounded-2xl text-[10px] font-black text-[#00ff88] uppercase tracking-[0.2em] hover:bg-[#00ff88]/10 transition-all shadow-lg text-center"
                    >
                        <Sparkles size={18} />
                        重新分析並合成新方案
                    </button>
                </div>

                <button onClick={() => nav("/")} className="w-full bg-[#00ff88] text-[#0f2e24] py-5 rounded-2xl font-black text-sm uppercase shadow-lg flex items-center justify-center gap-3 mt-4"><Sparkles size={20} />完成並存檔</button>
            </div>
        </div>
    );
}
