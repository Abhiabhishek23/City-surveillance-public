"""add latitude and longitude to cameras

Revision ID: 20250916_01
Revises: 
Create Date: 2025-09-16 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '20250916_01'
down_revision = None
branch_labels = None
depends_on = None

def upgrade():
    # Add columns if not exist (SQLite lacks IF NOT EXISTS for ALTER TABLE ADD COLUMN, but it's safe to attempt once)
    with op.batch_alter_table('cameras') as batch_op:
        batch_op.add_column(sa.Column('latitude', sa.Float(), nullable=True))
        batch_op.add_column(sa.Column('longitude', sa.Float(), nullable=True))


def downgrade():
    with op.batch_alter_table('cameras') as batch_op:
        batch_op.drop_column('longitude')
        batch_op.drop_column('latitude')
