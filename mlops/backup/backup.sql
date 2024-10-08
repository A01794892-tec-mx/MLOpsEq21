PGDMP  3    #             	    |            mlflow    17.0 (Debian 17.0-1.pgdg120+1)    17.0 (Debian 17.0-1.pgdg120+1) [    �           0    0    ENCODING    ENCODING        SET client_encoding = 'UTF8';
                           false            �           0    0 
   STDSTRINGS 
   STDSTRINGS     (   SET standard_conforming_strings = 'on';
                           false            �           0    0 
   SEARCHPATH 
   SEARCHPATH     8   SELECT pg_catalog.set_config('search_path', '', false);
                           false            �           1262    16384    mlflow    DATABASE     q   CREATE DATABASE mlflow WITH TEMPLATE = template0 ENCODING = 'UTF8' LOCALE_PROVIDER = libc LOCALE = 'en_US.utf8';
    DROP DATABASE mlflow;
                     mlflow    false            �            1259    16446    alembic_version    TABLE     X   CREATE TABLE public.alembic_version (
    version_num character varying(32) NOT NULL
);
 #   DROP TABLE public.alembic_version;
       public         heap r       mlflow    false            �            1259    16550    datasets    TABLE     X  CREATE TABLE public.datasets (
    dataset_uuid character varying(36) NOT NULL,
    experiment_id integer NOT NULL,
    name character varying(500) NOT NULL,
    digest character varying(36) NOT NULL,
    dataset_source_type character varying(36) NOT NULL,
    dataset_source text NOT NULL,
    dataset_schema text,
    dataset_profile text
);
    DROP TABLE public.datasets;
       public         heap r       mlflow    false            �            1259    16459    experiment_tags    TABLE     �   CREATE TABLE public.experiment_tags (
    key character varying(250) NOT NULL,
    value character varying(5000),
    experiment_id integer NOT NULL
);
 #   DROP TABLE public.experiment_tags;
       public         heap r       mlflow    false            �            1259    16386    experiments    TABLE     �  CREATE TABLE public.experiments (
    experiment_id integer NOT NULL,
    name character varying(256) NOT NULL,
    artifact_location character varying(256),
    lifecycle_stage character varying(32),
    creation_time bigint,
    last_update_time bigint,
    CONSTRAINT experiments_lifecycle_stage CHECK (((lifecycle_stage)::text = ANY ((ARRAY['active'::character varying, 'deleted'::character varying])::text[])))
);
    DROP TABLE public.experiments;
       public         heap r       mlflow    false            �            1259    16385    experiments_experiment_id_seq    SEQUENCE     �   CREATE SEQUENCE public.experiments_experiment_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;
 4   DROP SEQUENCE public.experiments_experiment_id_seq;
       public               mlflow    false    218            �           0    0    experiments_experiment_id_seq    SEQUENCE OWNED BY     _   ALTER SEQUENCE public.experiments_experiment_id_seq OWNED BY public.experiments.experiment_id;
          public               mlflow    false    217            �            1259    16571 
   input_tags    TABLE     �   CREATE TABLE public.input_tags (
    input_uuid character varying(36) NOT NULL,
    name character varying(255) NOT NULL,
    value character varying(500) NOT NULL
);
    DROP TABLE public.input_tags;
       public         heap r       mlflow    false            �            1259    16564    inputs    TABLE       CREATE TABLE public.inputs (
    input_uuid character varying(36) NOT NULL,
    source_type character varying(36) NOT NULL,
    source_id character varying(36) NOT NULL,
    destination_type character varying(36) NOT NULL,
    destination_id character varying(36) NOT NULL
);
    DROP TABLE public.inputs;
       public         heap r       mlflow    false            �            1259    16471    latest_metrics    TABLE     �   CREATE TABLE public.latest_metrics (
    key character varying(250) NOT NULL,
    value double precision NOT NULL,
    "timestamp" bigint,
    step bigint NOT NULL,
    is_nan boolean NOT NULL,
    run_uuid character varying(32) NOT NULL
);
 "   DROP TABLE public.latest_metrics;
       public         heap r       mlflow    false            �            1259    16424    metrics    TABLE       CREATE TABLE public.metrics (
    key character varying(250) NOT NULL,
    value double precision NOT NULL,
    "timestamp" bigint NOT NULL,
    run_uuid character varying(32) NOT NULL,
    step bigint DEFAULT '0'::bigint NOT NULL,
    is_nan boolean DEFAULT false NOT NULL
);
    DROP TABLE public.metrics;
       public         heap r       mlflow    false            �            1259    16519    model_version_tags    TABLE     �   CREATE TABLE public.model_version_tags (
    key character varying(250) NOT NULL,
    value character varying(5000),
    name character varying(256) NOT NULL,
    version integer NOT NULL
);
 &   DROP TABLE public.model_version_tags;
       public         heap r       mlflow    false            �            1259    16488    model_versions    TABLE       CREATE TABLE public.model_versions (
    name character varying(256) NOT NULL,
    version integer NOT NULL,
    creation_time bigint,
    last_updated_time bigint,
    description character varying(5000),
    user_id character varying(256),
    current_stage character varying(20),
    source character varying(500),
    run_id character varying(32),
    status character varying(20),
    status_message character varying(500),
    run_link character varying(500),
    storage_location character varying(500)
);
 "   DROP TABLE public.model_versions;
       public         heap r       mlflow    false            �            1259    16434    params    TABLE     �   CREATE TABLE public.params (
    key character varying(250) NOT NULL,
    value character varying(8000) NOT NULL,
    run_uuid character varying(32) NOT NULL
);
    DROP TABLE public.params;
       public         heap r       mlflow    false            �            1259    16538    registered_model_aliases    TABLE     �   CREATE TABLE public.registered_model_aliases (
    alias character varying(256) NOT NULL,
    version integer NOT NULL,
    name character varying(256) NOT NULL
);
 ,   DROP TABLE public.registered_model_aliases;
       public         heap r       mlflow    false            �            1259    16507    registered_model_tags    TABLE     �   CREATE TABLE public.registered_model_tags (
    key character varying(250) NOT NULL,
    value character varying(5000),
    name character varying(256) NOT NULL
);
 )   DROP TABLE public.registered_model_tags;
       public         heap r       mlflow    false            �            1259    16481    registered_models    TABLE     �   CREATE TABLE public.registered_models (
    name character varying(256) NOT NULL,
    creation_time bigint,
    last_updated_time bigint,
    description character varying(5000)
);
 %   DROP TABLE public.registered_models;
       public         heap r       mlflow    false            �            1259    16397    runs    TABLE     p  CREATE TABLE public.runs (
    run_uuid character varying(32) NOT NULL,
    name character varying(250),
    source_type character varying(20),
    source_name character varying(500),
    entry_point_name character varying(50),
    user_id character varying(256),
    status character varying(9),
    start_time bigint,
    end_time bigint,
    source_version character varying(50),
    lifecycle_stage character varying(20),
    artifact_uri character varying(200),
    experiment_id integer,
    deleted_time bigint,
    CONSTRAINT runs_lifecycle_stage CHECK (((lifecycle_stage)::text = ANY ((ARRAY['active'::character varying, 'deleted'::character varying])::text[]))),
    CONSTRAINT runs_status_check CHECK (((status)::text = ANY ((ARRAY['SCHEDULED'::character varying, 'FAILED'::character varying, 'FINISHED'::character varying, 'RUNNING'::character varying, 'KILLED'::character varying])::text[]))),
    CONSTRAINT source_type CHECK (((source_type)::text = ANY ((ARRAY['NOTEBOOK'::character varying, 'JOB'::character varying, 'LOCAL'::character varying, 'UNKNOWN'::character varying, 'PROJECT'::character varying])::text[])))
);
    DROP TABLE public.runs;
       public         heap r       mlflow    false            �            1259    16412    tags    TABLE     �   CREATE TABLE public.tags (
    key character varying(250) NOT NULL,
    value character varying(5000),
    run_uuid character varying(32) NOT NULL
);
    DROP TABLE public.tags;
       public         heap r       mlflow    false            �            1259    16578 
   trace_info    TABLE     �   CREATE TABLE public.trace_info (
    request_id character varying(50) NOT NULL,
    experiment_id integer NOT NULL,
    timestamp_ms bigint NOT NULL,
    execution_time_ms bigint,
    status character varying(50) NOT NULL
);
    DROP TABLE public.trace_info;
       public         heap r       mlflow    false            �            1259    16602    trace_request_metadata    TABLE     �   CREATE TABLE public.trace_request_metadata (
    key character varying(250) NOT NULL,
    value character varying(8000),
    request_id character varying(50) NOT NULL
);
 *   DROP TABLE public.trace_request_metadata;
       public         heap r       mlflow    false            �            1259    16589 
   trace_tags    TABLE     �   CREATE TABLE public.trace_tags (
    key character varying(250) NOT NULL,
    value character varying(8000),
    request_id character varying(50) NOT NULL
);
    DROP TABLE public.trace_tags;
       public         heap r       mlflow    false            �           2604    16630    experiments experiment_id    DEFAULT     �   ALTER TABLE ONLY public.experiments ALTER COLUMN experiment_id SET DEFAULT nextval('public.experiments_experiment_id_seq'::regclass);
 H   ALTER TABLE public.experiments ALTER COLUMN experiment_id DROP DEFAULT;
       public               mlflow    false    218    217    218            �          0    16446    alembic_version 
   TABLE DATA           6   COPY public.alembic_version (version_num) FROM stdin;
    public               mlflow    false    223   �|       �          0    16550    datasets 
   TABLE DATA           �   COPY public.datasets (dataset_uuid, experiment_id, name, digest, dataset_source_type, dataset_source, dataset_schema, dataset_profile) FROM stdin;
    public               mlflow    false    231   �|       �          0    16459    experiment_tags 
   TABLE DATA           D   COPY public.experiment_tags (key, value, experiment_id) FROM stdin;
    public               mlflow    false    224   �|       �          0    16386    experiments 
   TABLE DATA              COPY public.experiments (experiment_id, name, artifact_location, lifecycle_stage, creation_time, last_update_time) FROM stdin;
    public               mlflow    false    218   �|       �          0    16571 
   input_tags 
   TABLE DATA           =   COPY public.input_tags (input_uuid, name, value) FROM stdin;
    public               mlflow    false    233   z}       �          0    16564    inputs 
   TABLE DATA           f   COPY public.inputs (input_uuid, source_type, source_id, destination_type, destination_id) FROM stdin;
    public               mlflow    false    232   �}       �          0    16471    latest_metrics 
   TABLE DATA           Y   COPY public.latest_metrics (key, value, "timestamp", step, is_nan, run_uuid) FROM stdin;
    public               mlflow    false    225   �}       �          0    16424    metrics 
   TABLE DATA           R   COPY public.metrics (key, value, "timestamp", run_uuid, step, is_nan) FROM stdin;
    public               mlflow    false    221   >~       �          0    16519    model_version_tags 
   TABLE DATA           G   COPY public.model_version_tags (key, value, name, version) FROM stdin;
    public               mlflow    false    229   �~       �          0    16488    model_versions 
   TABLE DATA           �   COPY public.model_versions (name, version, creation_time, last_updated_time, description, user_id, current_stage, source, run_id, status, status_message, run_link, storage_location) FROM stdin;
    public               mlflow    false    227   �~       �          0    16434    params 
   TABLE DATA           6   COPY public.params (key, value, run_uuid) FROM stdin;
    public               mlflow    false    222          �          0    16538    registered_model_aliases 
   TABLE DATA           H   COPY public.registered_model_aliases (alias, version, name) FROM stdin;
    public               mlflow    false    230   �       �          0    16507    registered_model_tags 
   TABLE DATA           A   COPY public.registered_model_tags (key, value, name) FROM stdin;
    public               mlflow    false    228   �       �          0    16481    registered_models 
   TABLE DATA           `   COPY public.registered_models (name, creation_time, last_updated_time, description) FROM stdin;
    public               mlflow    false    226   �       �          0    16397    runs 
   TABLE DATA           �   COPY public.runs (run_uuid, name, source_type, source_name, entry_point_name, user_id, status, start_time, end_time, source_version, lifecycle_stage, artifact_uri, experiment_id, deleted_time) FROM stdin;
    public               mlflow    false    219   �       �          0    16412    tags 
   TABLE DATA           4   COPY public.tags (key, value, run_uuid) FROM stdin;
    public               mlflow    false    220   ��       �          0    16578 
   trace_info 
   TABLE DATA           h   COPY public.trace_info (request_id, experiment_id, timestamp_ms, execution_time_ms, status) FROM stdin;
    public               mlflow    false    234    �       �          0    16602    trace_request_metadata 
   TABLE DATA           H   COPY public.trace_request_metadata (key, value, request_id) FROM stdin;
    public               mlflow    false    236   �       �          0    16589 
   trace_tags 
   TABLE DATA           <   COPY public.trace_tags (key, value, request_id) FROM stdin;
    public               mlflow    false    235   :�       �           0    0    experiments_experiment_id_seq    SEQUENCE SET     K   SELECT pg_catalog.setval('public.experiments_experiment_id_seq', 2, true);
          public               mlflow    false    217            �           2606    16450 #   alembic_version alembic_version_pkc 
   CONSTRAINT     j   ALTER TABLE ONLY public.alembic_version
    ADD CONSTRAINT alembic_version_pkc PRIMARY KEY (version_num);
 M   ALTER TABLE ONLY public.alembic_version DROP CONSTRAINT alembic_version_pkc;
       public                 mlflow    false    223            �           2606    16556    datasets dataset_pk 
   CONSTRAINT     j   ALTER TABLE ONLY public.datasets
    ADD CONSTRAINT dataset_pk PRIMARY KEY (experiment_id, name, digest);
 =   ALTER TABLE ONLY public.datasets DROP CONSTRAINT dataset_pk;
       public                 mlflow    false    231    231    231            �           2606    16394    experiments experiment_pk 
   CONSTRAINT     b   ALTER TABLE ONLY public.experiments
    ADD CONSTRAINT experiment_pk PRIMARY KEY (experiment_id);
 C   ALTER TABLE ONLY public.experiments DROP CONSTRAINT experiment_pk;
       public                 mlflow    false    218            �           2606    16465 !   experiment_tags experiment_tag_pk 
   CONSTRAINT     o   ALTER TABLE ONLY public.experiment_tags
    ADD CONSTRAINT experiment_tag_pk PRIMARY KEY (key, experiment_id);
 K   ALTER TABLE ONLY public.experiment_tags DROP CONSTRAINT experiment_tag_pk;
       public                 mlflow    false    224    224            �           2606    16396     experiments experiments_name_key 
   CONSTRAINT     [   ALTER TABLE ONLY public.experiments
    ADD CONSTRAINT experiments_name_key UNIQUE (name);
 J   ALTER TABLE ONLY public.experiments DROP CONSTRAINT experiments_name_key;
       public                 mlflow    false    218                       2606    16577    input_tags input_tags_pk 
   CONSTRAINT     d   ALTER TABLE ONLY public.input_tags
    ADD CONSTRAINT input_tags_pk PRIMARY KEY (input_uuid, name);
 B   ALTER TABLE ONLY public.input_tags DROP CONSTRAINT input_tags_pk;
       public                 mlflow    false    233    233                        2606    16568    inputs inputs_pk 
   CONSTRAINT     �   ALTER TABLE ONLY public.inputs
    ADD CONSTRAINT inputs_pk PRIMARY KEY (source_type, source_id, destination_type, destination_id);
 :   ALTER TABLE ONLY public.inputs DROP CONSTRAINT inputs_pk;
       public                 mlflow    false    232    232    232    232            �           2606    16475    latest_metrics latest_metric_pk 
   CONSTRAINT     h   ALTER TABLE ONLY public.latest_metrics
    ADD CONSTRAINT latest_metric_pk PRIMARY KEY (key, run_uuid);
 I   ALTER TABLE ONLY public.latest_metrics DROP CONSTRAINT latest_metric_pk;
       public                 mlflow    false    225    225            �           2606    16533    metrics metric_pk 
   CONSTRAINT     |   ALTER TABLE ONLY public.metrics
    ADD CONSTRAINT metric_pk PRIMARY KEY (key, "timestamp", step, run_uuid, value, is_nan);
 ;   ALTER TABLE ONLY public.metrics DROP CONSTRAINT metric_pk;
       public                 mlflow    false    221    221    221    221    221    221            �           2606    16494    model_versions model_version_pk 
   CONSTRAINT     h   ALTER TABLE ONLY public.model_versions
    ADD CONSTRAINT model_version_pk PRIMARY KEY (name, version);
 I   ALTER TABLE ONLY public.model_versions DROP CONSTRAINT model_version_pk;
       public                 mlflow    false    227    227            �           2606    16525 '   model_version_tags model_version_tag_pk 
   CONSTRAINT     u   ALTER TABLE ONLY public.model_version_tags
    ADD CONSTRAINT model_version_tag_pk PRIMARY KEY (key, name, version);
 Q   ALTER TABLE ONLY public.model_version_tags DROP CONSTRAINT model_version_tag_pk;
       public                 mlflow    false    229    229    229            �           2606    16440    params param_pk 
   CONSTRAINT     X   ALTER TABLE ONLY public.params
    ADD CONSTRAINT param_pk PRIMARY KEY (key, run_uuid);
 9   ALTER TABLE ONLY public.params DROP CONSTRAINT param_pk;
       public                 mlflow    false    222    222            �           2606    16544 2   registered_model_aliases registered_model_alias_pk 
   CONSTRAINT     y   ALTER TABLE ONLY public.registered_model_aliases
    ADD CONSTRAINT registered_model_alias_pk PRIMARY KEY (name, alias);
 \   ALTER TABLE ONLY public.registered_model_aliases DROP CONSTRAINT registered_model_alias_pk;
       public                 mlflow    false    230    230            �           2606    16487 %   registered_models registered_model_pk 
   CONSTRAINT     e   ALTER TABLE ONLY public.registered_models
    ADD CONSTRAINT registered_model_pk PRIMARY KEY (name);
 O   ALTER TABLE ONLY public.registered_models DROP CONSTRAINT registered_model_pk;
       public                 mlflow    false    226            �           2606    16513 -   registered_model_tags registered_model_tag_pk 
   CONSTRAINT     r   ALTER TABLE ONLY public.registered_model_tags
    ADD CONSTRAINT registered_model_tag_pk PRIMARY KEY (key, name);
 W   ALTER TABLE ONLY public.registered_model_tags DROP CONSTRAINT registered_model_tag_pk;
       public                 mlflow    false    228    228            �           2606    16406    runs run_pk 
   CONSTRAINT     O   ALTER TABLE ONLY public.runs
    ADD CONSTRAINT run_pk PRIMARY KEY (run_uuid);
 5   ALTER TABLE ONLY public.runs DROP CONSTRAINT run_pk;
       public                 mlflow    false    219            �           2606    16418    tags tag_pk 
   CONSTRAINT     T   ALTER TABLE ONLY public.tags
    ADD CONSTRAINT tag_pk PRIMARY KEY (key, run_uuid);
 5   ALTER TABLE ONLY public.tags DROP CONSTRAINT tag_pk;
       public                 mlflow    false    220    220                       2606    16582    trace_info trace_info_pk 
   CONSTRAINT     ^   ALTER TABLE ONLY public.trace_info
    ADD CONSTRAINT trace_info_pk PRIMARY KEY (request_id);
 B   ALTER TABLE ONLY public.trace_info DROP CONSTRAINT trace_info_pk;
       public                 mlflow    false    234                       2606    16608 0   trace_request_metadata trace_request_metadata_pk 
   CONSTRAINT     {   ALTER TABLE ONLY public.trace_request_metadata
    ADD CONSTRAINT trace_request_metadata_pk PRIMARY KEY (key, request_id);
 Z   ALTER TABLE ONLY public.trace_request_metadata DROP CONSTRAINT trace_request_metadata_pk;
       public                 mlflow    false    236    236                       2606    16595    trace_tags trace_tag_pk 
   CONSTRAINT     b   ALTER TABLE ONLY public.trace_tags
    ADD CONSTRAINT trace_tag_pk PRIMARY KEY (key, request_id);
 A   ALTER TABLE ONLY public.trace_tags DROP CONSTRAINT trace_tag_pk;
       public                 mlflow    false    235    235            �           1259    16562    index_datasets_dataset_uuid    INDEX     X   CREATE INDEX index_datasets_dataset_uuid ON public.datasets USING btree (dataset_uuid);
 /   DROP INDEX public.index_datasets_dataset_uuid;
       public                 mlflow    false    231            �           1259    16563 0   index_datasets_experiment_id_dataset_source_type    INDEX     �   CREATE INDEX index_datasets_experiment_id_dataset_source_type ON public.datasets USING btree (experiment_id, dataset_source_type);
 D   DROP INDEX public.index_datasets_experiment_id_dataset_source_type;
       public                 mlflow    false    231    231            �           1259    16569 8   index_inputs_destination_type_destination_id_source_type    INDEX     �   CREATE INDEX index_inputs_destination_type_destination_id_source_type ON public.inputs USING btree (destination_type, destination_id, source_type);
 L   DROP INDEX public.index_inputs_destination_type_destination_id_source_type;
       public                 mlflow    false    232    232    232            �           1259    16570    index_inputs_input_uuid    INDEX     P   CREATE INDEX index_inputs_input_uuid ON public.inputs USING btree (input_uuid);
 +   DROP INDEX public.index_inputs_input_uuid;
       public                 mlflow    false    232            �           1259    16536    index_latest_metrics_run_uuid    INDEX     \   CREATE INDEX index_latest_metrics_run_uuid ON public.latest_metrics USING btree (run_uuid);
 1   DROP INDEX public.index_latest_metrics_run_uuid;
       public                 mlflow    false    225            �           1259    16535    index_metrics_run_uuid    INDEX     N   CREATE INDEX index_metrics_run_uuid ON public.metrics USING btree (run_uuid);
 *   DROP INDEX public.index_metrics_run_uuid;
       public                 mlflow    false    221            �           1259    16534    index_params_run_uuid    INDEX     L   CREATE INDEX index_params_run_uuid ON public.params USING btree (run_uuid);
 )   DROP INDEX public.index_params_run_uuid;
       public                 mlflow    false    222            �           1259    16537    index_tags_run_uuid    INDEX     H   CREATE INDEX index_tags_run_uuid ON public.tags USING btree (run_uuid);
 '   DROP INDEX public.index_tags_run_uuid;
       public                 mlflow    false    220                       1259    16588 +   index_trace_info_experiment_id_timestamp_ms    INDEX     y   CREATE INDEX index_trace_info_experiment_id_timestamp_ms ON public.trace_info USING btree (experiment_id, timestamp_ms);
 ?   DROP INDEX public.index_trace_info_experiment_id_timestamp_ms;
       public                 mlflow    false    234    234            	           1259    16614 '   index_trace_request_metadata_request_id    INDEX     p   CREATE INDEX index_trace_request_metadata_request_id ON public.trace_request_metadata USING btree (request_id);
 ;   DROP INDEX public.index_trace_request_metadata_request_id;
       public                 mlflow    false    236                       1259    16601    index_trace_tags_request_id    INDEX     X   CREATE INDEX index_trace_tags_request_id ON public.trace_tags USING btree (request_id);
 /   DROP INDEX public.index_trace_tags_request_id;
       public                 mlflow    false    235                       2606    16557 $   datasets datasets_experiment_id_fkey    FK CONSTRAINT     �   ALTER TABLE ONLY public.datasets
    ADD CONSTRAINT datasets_experiment_id_fkey FOREIGN KEY (experiment_id) REFERENCES public.experiments(experiment_id);
 N   ALTER TABLE ONLY public.datasets DROP CONSTRAINT datasets_experiment_id_fkey;
       public               mlflow    false    3290    231    218                       2606    16466 2   experiment_tags experiment_tags_experiment_id_fkey    FK CONSTRAINT     �   ALTER TABLE ONLY public.experiment_tags
    ADD CONSTRAINT experiment_tags_experiment_id_fkey FOREIGN KEY (experiment_id) REFERENCES public.experiments(experiment_id);
 \   ALTER TABLE ONLY public.experiment_tags DROP CONSTRAINT experiment_tags_experiment_id_fkey;
       public               mlflow    false    224    218    3290                       2606    16583 &   trace_info fk_trace_info_experiment_id    FK CONSTRAINT     �   ALTER TABLE ONLY public.trace_info
    ADD CONSTRAINT fk_trace_info_experiment_id FOREIGN KEY (experiment_id) REFERENCES public.experiments(experiment_id);
 P   ALTER TABLE ONLY public.trace_info DROP CONSTRAINT fk_trace_info_experiment_id;
       public               mlflow    false    3290    234    218                       2606    16620 ;   trace_request_metadata fk_trace_request_metadata_request_id    FK CONSTRAINT     �   ALTER TABLE ONLY public.trace_request_metadata
    ADD CONSTRAINT fk_trace_request_metadata_request_id FOREIGN KEY (request_id) REFERENCES public.trace_info(request_id) ON DELETE CASCADE;
 e   ALTER TABLE ONLY public.trace_request_metadata DROP CONSTRAINT fk_trace_request_metadata_request_id;
       public               mlflow    false    236    3333    234                       2606    16615 #   trace_tags fk_trace_tags_request_id    FK CONSTRAINT     �   ALTER TABLE ONLY public.trace_tags
    ADD CONSTRAINT fk_trace_tags_request_id FOREIGN KEY (request_id) REFERENCES public.trace_info(request_id) ON DELETE CASCADE;
 M   ALTER TABLE ONLY public.trace_tags DROP CONSTRAINT fk_trace_tags_request_id;
       public               mlflow    false    235    3333    234                       2606    16476 +   latest_metrics latest_metrics_run_uuid_fkey    FK CONSTRAINT     �   ALTER TABLE ONLY public.latest_metrics
    ADD CONSTRAINT latest_metrics_run_uuid_fkey FOREIGN KEY (run_uuid) REFERENCES public.runs(run_uuid);
 U   ALTER TABLE ONLY public.latest_metrics DROP CONSTRAINT latest_metrics_run_uuid_fkey;
       public               mlflow    false    3294    219    225                       2606    16429    metrics metrics_run_uuid_fkey    FK CONSTRAINT     �   ALTER TABLE ONLY public.metrics
    ADD CONSTRAINT metrics_run_uuid_fkey FOREIGN KEY (run_uuid) REFERENCES public.runs(run_uuid);
 G   ALTER TABLE ONLY public.metrics DROP CONSTRAINT metrics_run_uuid_fkey;
       public               mlflow    false    3294    219    221                       2606    16526 7   model_version_tags model_version_tags_name_version_fkey    FK CONSTRAINT     �   ALTER TABLE ONLY public.model_version_tags
    ADD CONSTRAINT model_version_tags_name_version_fkey FOREIGN KEY (name, version) REFERENCES public.model_versions(name, version) ON UPDATE CASCADE;
 a   ALTER TABLE ONLY public.model_version_tags DROP CONSTRAINT model_version_tags_name_version_fkey;
       public               mlflow    false    227    229    227    229    3314                       2606    16495 '   model_versions model_versions_name_fkey    FK CONSTRAINT     �   ALTER TABLE ONLY public.model_versions
    ADD CONSTRAINT model_versions_name_fkey FOREIGN KEY (name) REFERENCES public.registered_models(name) ON UPDATE CASCADE;
 Q   ALTER TABLE ONLY public.model_versions DROP CONSTRAINT model_versions_name_fkey;
       public               mlflow    false    227    226    3312                       2606    16441    params params_run_uuid_fkey    FK CONSTRAINT     �   ALTER TABLE ONLY public.params
    ADD CONSTRAINT params_run_uuid_fkey FOREIGN KEY (run_uuid) REFERENCES public.runs(run_uuid);
 E   ALTER TABLE ONLY public.params DROP CONSTRAINT params_run_uuid_fkey;
       public               mlflow    false    219    3294    222                       2606    16545 9   registered_model_aliases registered_model_alias_name_fkey    FK CONSTRAINT     �   ALTER TABLE ONLY public.registered_model_aliases
    ADD CONSTRAINT registered_model_alias_name_fkey FOREIGN KEY (name) REFERENCES public.registered_models(name) ON UPDATE CASCADE ON DELETE CASCADE;
 c   ALTER TABLE ONLY public.registered_model_aliases DROP CONSTRAINT registered_model_alias_name_fkey;
       public               mlflow    false    226    230    3312                       2606    16514 5   registered_model_tags registered_model_tags_name_fkey    FK CONSTRAINT     �   ALTER TABLE ONLY public.registered_model_tags
    ADD CONSTRAINT registered_model_tags_name_fkey FOREIGN KEY (name) REFERENCES public.registered_models(name) ON UPDATE CASCADE;
 _   ALTER TABLE ONLY public.registered_model_tags DROP CONSTRAINT registered_model_tags_name_fkey;
       public               mlflow    false    226    228    3312                       2606    16407    runs runs_experiment_id_fkey    FK CONSTRAINT     �   ALTER TABLE ONLY public.runs
    ADD CONSTRAINT runs_experiment_id_fkey FOREIGN KEY (experiment_id) REFERENCES public.experiments(experiment_id);
 F   ALTER TABLE ONLY public.runs DROP CONSTRAINT runs_experiment_id_fkey;
       public               mlflow    false    218    3290    219                       2606    16419    tags tags_run_uuid_fkey    FK CONSTRAINT     |   ALTER TABLE ONLY public.tags
    ADD CONSTRAINT tags_run_uuid_fkey FOREIGN KEY (run_uuid) REFERENCES public.runs(run_uuid);
 A   ALTER TABLE ONLY public.tags DROP CONSTRAINT tags_run_uuid_fkey;
       public               mlflow    false    220    3294    219            �      x�3113501757I2����� #%T      �      x������ � �      �      x������ � �      �   s   x�m̱
� ����]R��;�:f�r�`�I_�]�~������`�1��)��b���z�ɠ��f��J��r�띡.�q��v<��Q��~���F˫�;������?cM<F      �      x������ � �      �      x������ � �      �   z   x�M̻�0E�ZƠ�(~J�F����1�nu�s��z��.t0	�޽K��j6�_��e�;2s-F���I	\���|�)X��������[c�� 1ns;�CyX��e��8j�#�(f      �   z   x�M̻�0E�ZƠ�(~J�F����1�nu�s��z��.t0	�޽K��j6�_����\���i�{ddRW�����|�)X������@Yc�� 1ns;�CyX��eί�8j�E(f      �      x������ � �      �      x������ � �      �   w   x���;�0��>r��wA��^?A��b#���@��S�:�>�����䜑L� ���Z�B�f�gm�q7�F���Z7�د����fU�r��2)�5Ĭ^j���_�C�Xk?�X<      �      x������ � �      �      x������ � �      �      x������ � �      �   �   x����J�@����U����w'��RAQ���̡�S*M0�o�W����-��T@fn�{j11�1��tC����
^�>>�E �\/��?6q�`�������C��j�7sy�H��}e8.�Q����4�w���F=�A�!V�1�����2�)��ƓJ�M	������G�'Ϳ�A��n�
�����-�/:��˪��fi�      �   0  x��S�n�0<�_a�\	��|�ܜ譭
aE.c�R��𿗒b mvP�֋@rgf��N��6O�У]��}Z}����~�/�QTJY�	�E!�౐BJ")�e3{3X�a.��,��o�g����?h�bY�=��V�e�+�A|�ؗ��{�-�J���ڰ;�Ձ;v��>|��^ʹC{?����Q_M��1h�@�T�=.��Z{�J��f��$���1�)	�U��H�����x�T��'EJhD$ ъƛ�m	IF✍��Z��=M��X� Oj�kA
� ��I@����`l������L[I��N�v:���v��^gQ(�[���n,hm���Ɖ������wnu@�ϝ�qEa1���L��i��i����u�u�p:yȹ�t)��NT��}�g�K#Q�t��U��O���6�&;�b�a����v��t:}�zt^ŭN�X
&�X/�(D��L��3������۵�s���=i����Y|
�^-q!r�d�!rtE�&�7�F�,{�s�}�h.�$��(Ҝ�����y�?r����S����7p/8n      �      x������ � �      �      x������ � �      �      x������ � �     