import { useEffect, useRef, useState } from "react";

export function useCamera() {
    const videoRef = useRef<HTMLVideoElement>(null);
    const streamRef = useRef<MediaStream | null>(null);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        async function setupCamera() {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({
                    video: { facingMode: "environment" },
                    audio: false
                });
                streamRef.current = stream;
                if (videoRef.current) {
                    videoRef.current.srcObject = stream;
                }
            } catch (err) {
                console.error("Error accessing camera:", err);
                setError("Camera access denied");
            }
        }

        setupCamera();

        return () => {
            if (streamRef.current) {
                streamRef.current.getTracks().forEach(track => track.stop());
            }
        };
    }, []);

    return { videoRef, error };
}
