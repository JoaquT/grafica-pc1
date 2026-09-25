import cv2
import numpy as np

video_path = 'video2.mp4'
cap = cv2.VideoCapture(video_path)

sustractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

# Contadores independientes y el total general
contador_derecha = 0
contador_izquierda = 0
contador_total = 0

# Lista de diccionarios para hacer seguimiento a cada persona de forma individual
rastreadores = [] 

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width = frame.shape[:2]

    # Recorte de la acera
    y_inicio = int(height * 0.55)
    frame_recortado = frame[y_inicio:height, 0:width]
    h_recorte, w_recorte = frame_recortado.shape[:2]
    
    linea_x = int(w_recorte * 0.5)

    # Procesamiento de imagen
    mascara = sustractor.apply(frame_recortado) 
    _, mascara_binaria = cv2.threshold(mascara, 200, 255, cv2.THRESH_BINARY) 
    
    # Supresion de ruido
    kernel_ruido = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)) 
    mascara_limpia = cv2.morphologyEx(mascara_binaria, cv2.MORPH_OPEN, kernel_ruido) 
    
    # Fusion de partes del cuerpo
    kernel_cuerpo = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 25)) 
    mascara_cerrada = cv2.morphologyEx(mascara_limpia, cv2.MORPH_CLOSE, kernel_cuerpo) 

    contornos, _ = cv2.findContours(mascara_cerrada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 

    color_linea = (0, 255, 255) 
    centroides_actuales = []

    for contorno in contornos:
        area = cv2.contourArea(contorno)
        if 1000 < area < 30000:
            x, y, w, h = cv2.boundingRect(contorno)
            relacion_aspecto = float(w) / h
            
            if 0.2 < relacion_aspecto < 1.3:
                cx = int(x + w / 2)
                cy = int(y + h / 2)
                centroides_actuales.append((cx, cy, x, y, w, h))

    # Logica de seguimiento
    for cx, cy, x, y, w, h in centroides_actuales:
        match = False
        
        for rastreador in rastreadores:
            distancia = np.hypot(cx - rastreador['cx'], cy - rastreador['cy'])
            
            if distancia < 30:
                match = True
                
                # REVISIÓN DE CRUCE DE LÍNEA EXACTO
                # Si estaba a la izquierda y ahora pasó a la derecha
                if rastreador['cx'] <= linea_x and cx > linea_x and not rastreador['contado']:
                    contador_derecha += 1
                    contador_total += 1
                    rastreador['contado'] = True
                    color_linea = (0, 255, 0) 
                
                # Si estaba a la derecha y ahora pasó a la izquierda
                elif rastreador['cx'] >= linea_x and cx < linea_x and not rastreador['contado']:
                    contador_izquierda += 1
                    contador_total += 1
                    rastreador['contado'] = True
                    color_linea = (0, 255, 0) 
                
                rastreador['cx'] = cx
                rastreador['cy'] = cy
                rastreador['estado'] = 10 
                break
        
        if not match:
            rastreadores.append({'cx': cx, 'cy': cy, 'contado': False, 'estado': 10})
        
        cv2.rectangle(frame_recortado, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.circle(frame_recortado, (cx, cy), 4, (0, 0, 255), -1)

    for rastreador in rastreadores:
        rastreador['estado'] -= 1
    rastreadores = [r for r in rastreadores if r['estado'] > 0]

    grosor_linea = 5 if color_linea == (0, 255, 0) else 2
    cv2.line(frame_recortado, (linea_x, 0), (linea_x, h_recorte), color_linea, grosor_linea)

    # Panel de resultados 
    cv2.rectangle(frame_recortado, (10, 10), (320, 115), (0, 0, 0), -1)
    cv2.putText(frame_recortado, f'Derecha ->: {contador_derecha}', (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame_recortado, f'<- Izquierda: {contador_izquierda}', (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame_recortado, f'Total: {contador_total}', (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow('Monitoreo con Tracking', frame_recortado)
    cv2.imshow('Mascara en Blanco y Negro (Filtro)', mascara_cerrada)
    if cv2.waitKey(16) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()