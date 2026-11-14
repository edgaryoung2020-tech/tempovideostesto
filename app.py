"""Local-only upload and review interface."""
import json
import os
import tempfile
from pathlib import Path
import streamlit as st
from engine import probe, transcribe, plan_overlays, validate_plan, render

st.set_page_config(page_title='Tempo · Video graphics',layout='wide')
st.title('Tempo / Motion overlays')
st.caption('Одно видео → смысловые акценты → анимированная графика. Исходное аудио сохраняется.')
with st.sidebar:
    st.subheader('Настройки')
    key=st.text_input('OpenAI API key',value=os.getenv('OPENAI_API_KEY',''),type='password')
    model=st.text_input('Модель OpenAI',value=os.getenv('OPENAI_MODEL',''))
    st.caption('Модель укажите из доступных в вашем API-аккаунте. Ключ не записывается в файлы.')
    whisper=st.selectbox('Распознавание речи',['small','medium','large-v3'])
    st.caption('Первое распознавание скачивает модель. Обработка идёт на компьютере, где запущено приложение.')
    consent=st.checkbox('Отправить расшифровку речи в OpenAI для смыслового анализа')
upload=st.file_uploader('Загрузить готовое видео',type=['mp4','mov','mkv'])
st.info('Стиль: тёмные карточки, светлая типографика, лаймовый акцент. Автоматический выбор учитывает речь; лица и объекты в кадре пока не анализируются.')
if upload is not None:
    identity=(upload.name,upload.size,upload.file_id)
    if st.session_state.get('identity')!=identity:
        old=st.session_state.get('work')
        if old: old.cleanup()
        work=tempfile.TemporaryDirectory(prefix='tempo-')
        source=Path(work.name)/('input'+Path(upload.name).suffix.lower())
        source.write_bytes(upload.getbuffer())
        st.session_state.update(identity=identity,work=work,source=source)
        for field in ('plan','output','editor'): st.session_state.pop(field,None)
    source=st.session_state['source']
    st.video(upload)
    try: duration=float(probe(source)['format']['duration'])
    except Exception:
        st.error('Не удалось прочитать видео. Проверьте файл и установку FFmpeg.'); st.stop()
    if st.button('Распознать речь и предложить вставки',type='primary'):
        if not key or not model or not consent:
            st.error('Укажите ключ, модель и разрешите отправку расшифровки. Либо загрузите готовый JSON-план ниже.')
        else:
            try:
                with st.spinner('Распознавание речи и выбор смысловых акцентов…'):
                    segments=transcribe(source,whisper)
                    plan=plan_overlays(segments,duration,key,model)
                    st.session_state['plan']=plan
                    st.session_state['editor']=json.dumps(plan,ensure_ascii=False,indent=2)
            except Exception as e: st.error(f'План не создан: {type(e).__name__}. Проверьте модель, доступ API и соединение.')
    plan_file=st.file_uploader('Или загрузить JSON-план без API',type=['json'])
    if plan_file and st.button('Применить загруженный план'):
        try:
            plan=validate_plan(json.loads(plan_file.getvalue()),duration)
            st.session_state['plan']=plan; st.session_state['editor']=json.dumps(plan,ensure_ascii=False,indent=2)
        except Exception as e: st.error(str(e))
    if 'plan' in st.session_state:
        st.subheader('Проверка текста и моментов появления')
        st.text_area('План вставок: секунды от начала видео',key='editor',height=320)
        if st.button('Собрать видео',type='primary'):
            st.session_state.pop('output',None)
            try:
                plan=validate_plan(json.loads(st.session_state['editor']),duration)
                output=Path(st.session_state['work'].name)/'tempo.mp4'
                bar=st.progress(0.)
                render(source,output,plan,bar.progress)
                st.session_state['output']=output
                st.session_state['plan']=plan
            except Exception as e: st.error(f'Сборка не завершена: {e}')
    if st.session_state.get('output'):
        output=st.session_state['output']; st.video(str(output))
        st.download_button('Скачать MP4',output.read_bytes(),file_name='tempo.mp4',mime='video/mp4')
        st.download_button('Скачать план',json.dumps(st.session_state['plan'],ensure_ascii=False,indent=2),file_name='tempo.plan.json')
