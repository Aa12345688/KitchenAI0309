import React, { createContext, useContext, useState, useEffect, ReactNode } from "react";
import { llmService } from "./llmService";

export interface ScannedItem {
    id: string;
    name: string;
    quantity: number;
    timestamp: number;
    category?: string;
    freshness?: number; // 0-10
    expiryDays?: number; // Days until expiry
    confidence?: number;
    isSpoiled?: boolean;
    box?: number[]; // [x1, y1, x2, y2]
    storageType?: "fridge" | "freezer";
}

export interface WasteRecord {
    date: string;
    amount: number; // in grams or items
}

interface IngredientContextType {
    scannedItems: ScannedItem[];
    recommendedRecipes: any[];
    tempDetections: ScannedItem[];
    selectedIds: string[];
    settings: { notifications: boolean; darkMode: boolean };
    wasteHistory: WasteRecord[];
    addItem: (item: Partial<ScannedItem>) => void;
    updateQuantity: (id: string, delta: number) => void;
    updateItem: (id: string, updates: Partial<ScannedItem>) => void;
    removeItem: (id: string) => void;
    removeIngredient: (id: string) => void;
    toggleSelection: (id: string) => void;
    generateRecipe: () => Promise<void>;
    clearAll: () => void;
    setRecipes: (recipes: any[]) => void;
    clearTempDetections: () => void;
    updateSettings: (settings: Partial<{ notifications: boolean; darkMode: boolean }>) => void;
}

const IngredientContext = createContext<IngredientContextType | undefined>(undefined);

export function IngredientProvider({ children }: { children: ReactNode }) {
    const [scannedItems, setScannedItems] = useState<ScannedItem[]>([]);
    const [recommendedRecipes, setRecommendedRecipes] = useState<any[]>([]);
    const [tempDetections, setTempDetections] = useState<ScannedItem[]>([]);
    const [selectedIds, setSelectedIds] = useState<string[]>([]);
    const [settings, setSettings] = useState({ notifications: true, darkMode: true });

    // Mock Waste History
    const [wasteHistory] = useState<WasteRecord[]>([
        { date: "2024-03-04", amount: 2 },
        { date: "2024-03-05", amount: 0 },
        { date: "2024-03-06", amount: 3 },
        { date: "2024-03-07", amount: 1 },
        { date: "2024-03-08", amount: 0 },
        { date: "2024-03-09", amount: 4 },
        { date: "2024-03-10", amount: 1 },
    ]);

    // Load from localStorage on mount
    useEffect(() => {
        const saved = localStorage.getItem("scannedIngredients");
        const savedRecipes = localStorage.getItem("recommendedRecipes");
        const savedSettings = localStorage.getItem("appSettings");
        if (saved) try { setScannedItems(JSON.parse(saved)); } catch (e) { }
        if (savedRecipes) try { setRecommendedRecipes(JSON.parse(savedRecipes)); } catch (e) { }
        if (savedSettings) try { setSettings(JSON.parse(savedSettings)); } catch (e) { }
    }, []);

    // Sync to localStorage
    useEffect(() => {
        localStorage.setItem("scannedIngredients", JSON.stringify(scannedItems));
    }, [scannedItems]);

    useEffect(() => {
        localStorage.setItem("recommendedRecipes", JSON.stringify(recommendedRecipes));
    }, [recommendedRecipes]);

    useEffect(() => {
        localStorage.setItem("appSettings", JSON.stringify(settings));
    }, [settings]);

    const addItem = (item: Partial<ScannedItem>) => {
        const now = Date.now();
        const uniqueId = item.id || `${now}-${Math.random().toString(36).substr(2, 9)}`;
        const newItem: ScannedItem = {
            id: uniqueId,
            name: item.name || "未知食材",
            quantity: item.quantity || 1,
            timestamp: item.timestamp || now,
            category: item.category || "其他",
            storageType: item.storageType || "fridge",
            expiryDays: item.expiryDays || 7, // Default 7 days
            ...item
        };

        // Don't add to main inventory if spoiled (or handle differently)
        // Here we add it but tag it as spoiled
        setScannedItems(prev => {
            const existing = prev.find(i => i.name.toLowerCase() === newItem.name.toLowerCase() && i.isSpoiled === newItem.isSpoiled && i.storageType === newItem.storageType);
            if (existing) {
                return prev.map(i =>
                    i.id === existing.id
                        ? { ...i, quantity: i.quantity + newItem.quantity, timestamp: now }
                        : i
                );
            }
            return [...prev, newItem];
        });

        // Update temp detections (real-time preview)
        setTempDetections(prev => {
            const existing = prev.find(i => i.name.toLowerCase() === newItem.name.toLowerCase() && i.isSpoiled === newItem.isSpoiled && i.storageType === newItem.storageType);
            if (existing) {
                return prev.map(i =>
                    i.id === existing.id
                        ? { ...i, quantity: i.quantity + newItem.quantity, timestamp: now }
                        : i
                );
            }
            return [...prev, newItem];
        });

        // Auto-select ONLY IF NOT SPOILED
        if (!newItem.isSpoiled) {
            setSelectedIds(prev => {
                const existingItem = scannedItems.find(i => i.name.toLowerCase() === newItem.name.toLowerCase() && i.isSpoiled === newItem.isSpoiled && i.storageType === newItem.storageType);
                const targetId = existingItem ? existingItem.id : uniqueId;
                return prev.includes(targetId) ? prev : [...prev, targetId];
            });
        }
    };

    const clearTempDetections = () => setTempDetections([]);

    const updateQuantity = (id: string, delta: number) => {
        const updater = (prev: ScannedItem[]) =>
            prev.map(item =>
                item.id === id ? { ...item, quantity: Math.max(0, item.quantity + delta) } : item
            ).filter(item => item.quantity > 0);

        setScannedItems(updater);
        setTempDetections(updater);
    };

    const updateItem = (id: string, updates: Partial<ScannedItem>) => {
        setScannedItems(prev => prev.map(item => item.id === id ? { ...item, ...updates } : item));
        setTempDetections(prev => prev.map(item => item.id === id ? { ...item, ...updates } : item));
    };

    const removeItem = (id: string) => {
        if (!id) return;
        setScannedItems(prev => prev.filter(item => item.id !== id));
        setTempDetections(prev => prev.filter(item => item.id !== id));
        setSelectedIds(prev => prev.filter(selectedId => selectedId !== id));
    };

    const removeIngredient = removeItem;

    const generateRecipe = async () => {
        const selectedIngredients = scannedItems
            .filter(item => selectedIds.includes(item.id) && !item.isSpoiled)
            .map(item => item.name);

        if (selectedIngredients.length === 0) {
            throw new Error("請選擇有效的食材進行合成（損壞食材將自動排除）");
        }

        const recipes = await llmService.generateRecipes({ ingredients: selectedIngredients });
        setRecommendedRecipes(recipes);
    };

    const toggleSelection = (id: string) => {
        const item = scannedItems.find(i => i.id === id);
        if (item?.isSpoiled) return; // Cannot select spoiled items
        setSelectedIds(prev =>
            prev.includes(id) ? prev.filter(i => i !== id) : [...prev, id]
        );
    };

    const clearAll = () => {
        setScannedItems([]);
        setRecommendedRecipes([]);
        setTempDetections([]);
    };

    const updateSettings = (newSettings: Partial<{ notifications: boolean }>) => {
        setSettings(prev => ({ ...prev, ...newSettings }));
    };

    return (
        <IngredientContext.Provider value={{
            scannedItems,
            recommendedRecipes,
            tempDetections,
            selectedIds,
            settings,
            wasteHistory,
            addItem,
            updateQuantity,
            updateItem,
            removeItem,
            removeIngredient,
            toggleSelection,
            generateRecipe,
            clearAll,
            setRecipes: setRecommendedRecipes,
            clearTempDetections,
            updateSettings
        }}>
            {children}
        </IngredientContext.Provider>
    );
}

export function useIngredients() {
    const context = useContext(IngredientContext);
    if (context === undefined) {
        throw new Error("useIngredients must be used within an IngredientProvider");
    }
    return context;
}
