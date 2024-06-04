from fastapi import FastAPI, Depends
from models import Message
import supabase

SUPABASE_URL = "https://fnjsdxnejydzzlievpie.supabase.co/"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImZuanNkeG5lanlkenpsaWV2cGllIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTcwMzIxMzAyOCwiZXhwIjoyMDE4Nzg5MDI4fQ.DBcvEFlnsh3jlLLDWNAE8BIgYaLAhO2sMBwTFvVx23c"
supabase_client = supabase.Client(SUPABASE_URL, SUPABASE_KEY)




def save_message(question_message : Message):

    
    supabase_client.table