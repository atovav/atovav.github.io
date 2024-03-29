(function () {
	// Create the connector object
	var myConnector = tableau.makeConnector();


	// Define the schema
	myConnector.getSchema = function (schemaCallback) {
		var inputForm = JSON.parse(tableau.connectionData),
			callbackTables = [];

		var activitiesCols = [
			// { id: 'id', dataType: tableau.dataTypeEnum.int },
			{ id: 'company_id', dataType: tableau.dataTypeEnum.int },
			{ id: 'user_id', dataType: tableau.dataTypeEnum.int },
			{ id: 'done', dataType: tableau.dataTypeEnum.bool },
			{ id: 'type', dataType: tableau.dataTypeEnum.string },
			{ id: 'reference_type', dataType: tableau.dataTypeEnum.string },
			{ id: 'reference_id', dataType: tableau.dataTypeEnum.int },
			{ id: 'due_date', dataType: tableau.dataTypeEnum.date },
			{ id: 'due_time', dataType: tableau.dataTypeEnum.string },
			{ id: 'duration', dataType: tableau.dataTypeEnum.string },
			{ id: 'add_time', dataType: tableau.dataTypeEnum.date },
			{ id: 'marked_as_done_time', dataType: tableau.dataTypeEnum.date },
			{ id: 'subject', dataType: tableau.dataTypeEnum.string },
			{ id: 'deal_id', dataType: tableau.dataTypeEnum.string },
			{ id: 'org_id', dataType: tableau.dataTypeEnum.string },
			{ id: 'person_id', dataType: tableau.dataTypeEnum.int },
			{ id: 'active_flag', dataType: tableau.dataTypeEnum.bool },
			{ id: 'update_time', dataType: tableau.dataTypeEnum.date },
			{ id: 'gcal_evevnt_id', dataType: tableau.dataTypeEnum.int },
			{ id: 'google_calendar_id', dataType: tableau.dataTypeEnum.int },
			{ id: 'google_calendar_etag', dataType: tableau.dataTypeEnum.string },
			{ id: 'note', dataType: tableau.dataTypeEnum.string },
			{ id: 'participant_count', dataType: tableau.dataTypeEnum.int },
			{ id: 'participants', dataType: tableau.dataTypeEnum.string },
			{ id: 'person_name', dataType: tableau.dataTypeEnum.string },
			{ id: 'org_name', dataType: tableau.dataTypeEnum.string },
			{ id: 'deal_title', dataType: tableau.dataTypeEnum.string },
			{ id: 'assigned_to_user_id', dataType: tableau.dataTypeEnum.int },
			{ id: 'created_by_user_id', dataType: tableau.dataTypeEnum.int },
			{ id: 'owner_name', dataType: tableau.dataTypeEnum.string },
			{ id: 'person_dropbox_bcc', dataType: tableau.dataTypeEnum.string },
			{ id: 'deal_dropbox_bcc', dataType: tableau.dataTypeEnum.string },
		];

		var dealsCols = [
			{ id: 'Título',dataType: tableau.dataTypeEnum.string },
			{ id: 'Creador',dataType: tableau.dataTypeEnum.string },
			{ id: 'Propietario',dataType: tableau.dataTypeEnum.string },
			{ id: 'Valor',dataType: tableau.dataTypeEnum.int },
			{ id: 'Embudo',dataType: tableau.dataTypeEnum.string },
			{ id: 'Nombre del Contacto',dataType: tableau.dataTypeEnum.string },
			{ id: 'Etapa',dataType: tableau.dataTypeEnum.string },
			{ id: 'Etiqueta',dataType: tableau.dataTypeEnum.string },
			{ id: 'Estado',dataType: tableau.dataTypeEnum.string },
			{ id: 'Trato creado',dataType: tableau.dataTypeEnum.date },
			{ id: 'Fecha de actualización',dataType: tableau.dataTypeEnum.date },
			{ id: 'Último cambio de etapa',dataType: tableau.dataTypeEnum.date },
			{ id: 'Fecha de próxima acción',dataType: tableau.dataTypeEnum.date },
			{ id: 'Fecha de última acción',dataType: tableau.dataTypeEnum.date },
			{ id: 'Fecha de ganado',dataType: tableau.dataTypeEnum.date },
			{ id: 'Último correo recibido',dataType: tableau.dataTypeEnum.date },
			{ id: 'Último correo enviado',dataType: tableau.dataTypeEnum.date },
			{ id: 'Fecha de perdido',dataType: tableau.dataTypeEnum.date },
			{ id: 'Trato cerrado el',dataType: tableau.dataTypeEnum.date },
			{ id: 'Motivo de pérdida',dataType: tableau.dataTypeEnum.string },
			{ id: 'Visible para',dataType: tableau.dataTypeEnum.string },
			{ id: 'ID',dataType: tableau.dataTypeEnum.int },
			{ id: 'Acciones totales',dataType: tableau.dataTypeEnum.int },
			{ id: 'Acciones completadas',dataType: tableau.dataTypeEnum.int },
			{ id: 'Acciones por hacer',dataType: tableau.dataTypeEnum.int },
			{ id: 'Número de mensajes de correo electrónico',dataType: tableau.dataTypeEnum.int },
			{ id: 'Dolores Principales',dataType: tableau.dataTypeEnum.string },
			{ id: '¿Estás de acuerdo con recibir por tu auto el monto que te ofrecemos??',dataType: tableau.dataTypeEnum.string },
			{ id: '¿El coche está a tu nombre?',dataType: tableau.dataTypeEnum.string },
			{ id: '¿El auto se ve, huele y funciona bien?',dataType: tableau.dataTypeEnum.string },
			{ id: '¿Tiene Crédito Vigente?',dataType: tableau.dataTypeEnum.string },
			{ id: '¿Te interesa publicar tu auto en Odetta para poder venderlo al mejor precio?',dataType: tableau.dataTypeEnum.string },
			{ id: '¿Cómo quieres mostrarnos tus documentos?',dataType: tableau.dataTypeEnum.string },
			{ id: '¿Quieres empezar el proceso de publicación?',dataType: tableau.dataTypeEnum.string },
			{ id: '¿Cuándo quieres comenzar el proceso de publicación?',dataType: tableau.dataTypeEnum.string },
			{ id: '¿En cuánto tiempo te gustaría publicar tu auto?',dataType: tableau.dataTypeEnum.string },
			{ id: '¿Cómo te gustaría que te contactemos?',dataType: tableau.dataTypeEnum.string },
			{ id: '¿Sabes de algún documento que no esté al corriente?',dataType: tableau.dataTypeEnum.string },
			{ id: 'Si fuera necesario, ¿Cómo te gustaría que actualicemos tus documentos?',dataType: tableau.dataTypeEnum.string },
			{ id: 'Si fuera necesario, ¿Cómo te gustaría reparar el auto?',dataType: tableau.dataTypeEnum.string },
			{ id: '¿El auto tiene seguro vigente?',dataType: tableau.dataTypeEnum.string },
			{ id: '¿Para qué parte del proceso te interesa el servicio de Valet?',dataType: tableau.dataTypeEnum.string },
			{ id: '¿A qué embudo pasa cuando está calificado?',dataType: tableau.dataTypeEnum.string },
			{ id: 'Incluyéndolo a usted, ¿Cuántos dueños ha tenido el auto?',dataType: tableau.dataTypeEnum.string },
			{ id: '¿Cómo compraste tu auto?',dataType: tableau.dataTypeEnum.string },
			{ id: '¿El auto ha sido reemplacado?',dataType: tableau.dataTypeEnum.string },
			{ id: '¿De qué estado son las placas del auto?',dataType: tableau.dataTypeEnum.string },
			{ id: '¿Ha sido modificado?',dataType: tableau.dataTypeEnum.string },
			{ id: '¿Tiene garantía de fábrica?',dataType: tableau.dataTypeEnum.string },
			{ id: '¿Cuál es el estado mecánico de tu auto?',dataType: tableau.dataTypeEnum.string },
			{ id: '¿Cuál es el estado estético del exterior de tu auto?',dataType: tableau.dataTypeEnum.string },
			{ id: '¿Cuál es el estado estético del interior de tu auto?',dataType: tableau.dataTypeEnum.string },
			{ id: 'Fecha de último servicio',dataType: tableau.dataTypeEnum.string },
			{ id: 'Certificaciones',dataType: tableau.dataTypeEnum.string },
			{ id: 'Marca',dataType: tableau.dataTypeEnum.string },
			{ id: 'Modelo',dataType: tableau.dataTypeEnum.string },
			{ id: 'Año',dataType: tableau.dataTypeEnum.int },
			{ id: 'Version',dataType: tableau.dataTypeEnum.string },
			{ id: 'Kilometraje',dataType: tableau.dataTypeEnum.string },
			{ id: 'Comisión Odetta',dataType: tableau.dataTypeEnum.int },
			{ id: 'SKU',dataType: tableau.dataTypeEnum.string },
			{ id: 'Pagable a Cliente',dataType: tableau.dataTypeEnum.int },
			{ id: 'Precio de Publicación',dataType: tableau.dataTypeEnum.int },
			{ id: 'Precio del Mercado',dataType: tableau.dataTypeEnum.int },
			{ id: 'Origen',dataType: tableau.dataTypeEnum.string },
			{ id: 'Originador',dataType: tableau.dataTypeEnum.string },
			{ id: 'UTUM_Source',dataType: tableau.dataTypeEnum.string },
			{ id: 'UTUM_Campaign',dataType: tableau.dataTypeEnum.string },
			{ id: 'UTUM_Medium',dataType: tableau.dataTypeEnum.string },
			{ id: 'MKT_Origen',dataType: tableau.dataTypeEnum.string },
			{ id: 'MKT_Originador',dataType: tableau.dataTypeEnum.string },
		];

		var goalsCols = [
			{ id: 'id', dataType: tableau.dataTypeEnum.int },
			{ id: 'company_id', dataType: tableau.dataTypeEnum.int },
			{ id: 'user_id', dataType: tableau.dataTypeEnum.int },
			{ id: 'stage_id', dataType: tableau.dataTypeEnum.int },
			{ id: 'active_goal_id', dataType: tableau.dataTypeEnum.int },
			{ id: 'period', dataType: tableau.dataTypeEnum.string },
			{ id: 'expected', dataType: tableau.dataTypeEnum.int },
			{ id: 'active_flag', dataType: tableau.dataTypeEnum.bool },
			{ id: 'add_time', dataType: tableau.dataTypeEnum.date },
			{ id: 'goal_type', dataType: tableau.dataTypeEnum.string },
			{ id: 'expected_sum', dataType: tableau.dataTypeEnum.int },
			{ id: 'currency', dataType: tableau.dataTypeEnum.string },
			{ id: 'expected_type', dataType: tableau.dataTypeEnum.string },
			{ id: 'created_by_user_id', dataType: tableau.dataTypeEnum.int },
			{ id: 'pipeline_id', dataType: tableau.dataTypeEnum.int },
			{ id: 'master_expected', dataType: tableau.dataTypeEnum.int },
			{ id: 'delivered', dataType: tableau.dataTypeEnum.int },
			{ id: 'delivered_sum', dataType: tableau.dataTypeEnum.int },
			{ id: 'period_start', dataType: tableau.dataTypeEnum.date },
			{ id: 'period_end', dataType: tableau.dataTypeEnum.date },
			{ id: 'user_name', dataType: tableau.dataTypeEnum.string },

		];

		var organizationsCols = [
			{ id: 'id', dataType: tableau.dataTypeEnum.int },
			{ id: 'company_id', dataType: tableau.dataTypeEnum.int },
			{ id: 'name', dataType: tableau.dataTypeEnum.string },
		];

		var notesCols = [
			{ id: 'id', dataType: tableau.dataTypeEnum.int },
			{ id: 'user_id', dataType: tableau.dataTypeEnum.int },
			{ id: 'deal_id', dataType: tableau.dataTypeEnum.int },
			{ id: 'person_id', dataType: tableau.dataTypeEnum.int },
			{ id: 'org_id', dataType: tableau.dataTypeEnum.int },
			{ id: 'content', dataType: tableau.dataTypeEnum.string },
			{ id: 'add_time', dataType: tableau.dataTypeEnum.date },
			{ id: 'update_time', dataType: tableau.dataTypeEnum.date },
			{ id: 'active_flag', dataType: tableau.dataTypeEnum.bool },
			{ id: 'pinned_to_deal_flag', dataType: tableau.dataTypeEnum.bool },
			{ id: 'pinned_to_person_flag', dataType: tableau.dataTypeEnum.bool },
			{ id: 'pinned_to_organization_flag', dataType: tableau.dataTypeEnum.bool },
			{ id: 'last_update_user_id', dataType: tableau.dataTypeEnum.int },
			{ id: 'organization', dataType: tableau.dataTypeEnum.int },
		];

		var personsCols = [
			{ id: 'id', dataType: tableau.dataTypeEnum.int },
			{ id: 'owner_id', dataType: tableau.dataTypeEnum.int },
			{ id: 'org_id', dataType: tableau.dataTypeEnum.int },
			{ id: 'first_name', dataType: tableau.dataTypeEnum.string },
			{ id: 'last_name', dataType: tableau.dataTypeEnum.string },
			{ id: 'open_deals_count', dataType: tableau.dataTypeEnum.int },
			{ id: 'related_open_deals_count', dataType: tableau.dataTypeEnum.int },
			{ id: 'closed_deals_count', dataType: tableau.dataTypeEnum.int },
			{ id: 'related_closed_deals_count', dataType: tableau.dataTypeEnum.int },
			{ id: 'participant_open_deals_count', dataType: tableau.dataTypeEnum.int },
			{ id: 'participant_closed_deals_count', dataType: tableau.dataTypeEnum.int },
			{ id: 'email_messages_count', dataType: tableau.dataTypeEnum.int },
			{ id: 'activities_count', dataType: tableau.dataTypeEnum.int },
			{ id: 'done_activities_count', dataType: tableau.dataTypeEnum.int },
			{ id: 'undone_activities_count', dataType: tableau.dataTypeEnum.int },
			{ id: 'reference_activities_count', dataType: tableau.dataTypeEnum.int },
			{ id: 'files_count', dataType: tableau.dataTypeEnum.int },
			{ id: 'notes_count', dataType: tableau.dataTypeEnum.int },
			{ id: 'followers_count', dataType: tableau.dataTypeEnum.int },
			{ id: 'won_deals_count', dataType: tableau.dataTypeEnum.int },
			{ id: 'related_won_deals_count', dataType: tableau.dataTypeEnum.int },
			{ id: 'lost_deals_count', dataType: tableau.dataTypeEnum.int },
			{ id: 'related_lost_deals_count', dataType: tableau.dataTypeEnum.int },
			{ id: 'active_flag', dataType: tableau.dataTypeEnum.bool },
			{ id: 'home_phone', dataType: tableau.dataTypeEnum.bool },
			{ id: 'work_phone', dataType: tableau.dataTypeEnum.bool },
			{ id: 'mobile_phone', dataType: tableau.dataTypeEnum.bool },
			{ id: 'other_phone', dataType: tableau.dataTypeEnum.bool },
			{ id: 'home_email', dataType: tableau.dataTypeEnum.bool },
			{ id: 'work_email', dataType: tableau.dataTypeEnum.bool },
			{ id: 'other_email', dataType: tableau.dataTypeEnum.bool },
			{ id: 'first_char', dataType: tableau.dataTypeEnum.string },
			{ id: 'update_time', dataType: tableau.dataTypeEnum.date },
			{ id: 'add_time', dataType: tableau.dataTypeEnum.date },
			{ id: 'visible_to', dataType: tableau.dataTypeEnum.string },
			{ id: 'picture_id', dataType: tableau.dataTypeEnum.int },
			{ id: 'next_activity_date', dataType: tableau.dataTypeEnum.date },
			{ id: 'next_activity_time', dataType: tableau.dataTypeEnum.date },
			{ id: 'next_activity_id', dataType: tableau.dataTypeEnum.int },
			{ id: 'last_activity_date', dataType: tableau.dataTypeEnum.date },
			{ id: 'last_activity_time', dataType: tableau.dataTypeEnum.date },
			{ id: 'last_activity_id', dataType: tableau.dataTypeEnum.int },
			{ id: 'last_incoming_mail_time', dataType: tableau.dataTypeEnum.date },
			{ id: 'last_outgoing_mail_time', dataType: tableau.dataTypeEnum.date },
			{ id: 'has_im', dataType: tableau.dataTypeEnum.bool },
			{ id: 'postal_address', dataType: tableau.dataTypeEnum.string },
			{ id: 'postal_address_subpremise', dataType: tableau.dataTypeEnum.string },
			{ id: 'postal_address_street_number', dataType: tableau.dataTypeEnum.string },
			{ id: 'postal_address_route', dataType: tableau.dataTypeEnum.string },
			{ id: 'postal_address_sublocality', dataType: tableau.dataTypeEnum.string },
			{ id: 'postal_address_locality', dataType: tableau.dataTypeEnum.string },
			{ id: 'postal_address_admin_area_level_1', dataType: tableau.dataTypeEnum.string },
			{ id: 'postal_address_admin_area_level_2', dataType: tableau.dataTypeEnum.string },
			{ id: 'postal_address_country', dataType: tableau.dataTypeEnum.string },
			{ id: 'postal_address_postal_code', dataType: tableau.dataTypeEnum.string },
			{ id: 'postal_address_formatted_address', dataType: tableau.dataTypeEnum.string },
			{ id: 'org_name', dataType: tableau.dataTypeEnum.string },
			{ id: 'cc_email', dataType: tableau.dataTypeEnum.string },
			{ id: 'owner_name', dataType: tableau.dataTypeEnum.string },
		];

		var productsCols = [
			{ id: 'id', dataType: tableau.dataTypeEnum.int },
			{ id: 'name', dataType: tableau.dataTypeEnum.string },
			{ id: 'code', dataType: tableau.dataTypeEnum.string },
			{ id: 'unit', dataType: tableau.dataTypeEnum.int },
			{ id: 'tax', dataType: tableau.dataTypeEnum.int },
			{ id: 'active_flag', dataType: tableau.dataTypeEnum.bool },
			{ id: 'selectable', dataType: tableau.dataTypeEnum.bool },
			{ id: 'first_char', dataType: tableau.dataTypeEnum.string },
			{ id: 'visible_to', dataType: tableau.dataTypeEnum.string },
			{ id: 'owner_id', dataType: tableau.dataTypeEnum.int },
			{ id: 'files_count', dataType: tableau.dataTypeEnum.int },
			{ id: 'followers_count', dataType: tableau.dataTypeEnum.date },
			{ id: 'add_time', dataType: tableau.dataTypeEnum.date },
			{ id: 'update_time', dataType: tableau.dataTypeEnum.date },
			{ id: 'price_amount', alias: 'price', dataType: tableau.dataTypeEnum.int },
			{ id: 'price_currency', dataType: tableau.dataTypeEnum.int },
			{ id: 'price_cost', dataType: tableau.dataTypeEnum.int },
			{ id: 'price_overhead_cost', dataType: tableau.dataTypeEnum.int },
			{ id: 'variation_id', dataType: tableau.dataTypeEnum.int },
			{ id: 'variation_name', dataType: tableau.dataTypeEnum.string },
			{ id: 'variation_price', dataType: tableau.dataTypeEnum.int },
			{ id: 'variation_comment', dataType: tableau.dataTypeEnum.string },

		];

		var stagesCols = [
			{ id: 'id', dataType: tableau.dataTypeEnum.int },
			{ id: 'order_nr', dataType: tableau.dataTypeEnum.int },
			{ id: 'name', dataType: tableau.dataTypeEnum.int },
			{ id: 'active_flag', dataType: tableau.dataTypeEnum.string },
			{ id: 'deal_probability', dataType: tableau.dataTypeEnum.int },
			{ id: 'pipeline_id', dataType: tableau.dataTypeEnum.int },
			{ id: 'rotten_flag', dataType: tableau.dataTypeEnum.bool },
			{ id: 'rotten_days', dataType: tableau.dataTypeEnum.int },
			{ id: 'add_time', dataType: tableau.dataTypeEnum.date },
			{ id: 'update_time', dataType: tableau.dataTypeEnum.date },
			{ id: 'pipeline_name', dataType: tableau.dataTypeEnum.date },
		];

		// Schema for users
		var usersCols = [
			{ id: 'id', dataType: tableau.dataTypeEnum.int },
			{ id: 'name', dataType: tableau.dataTypeEnum.string },
			{ id: 'default_currency', dataType: tableau.dataTypeEnum.string },
			{ id: 'locale', dataType: tableau.dataTypeEnum.string },
			{ id: 'lang', dataType: tableau.dataTypeEnum.int },
			{ id: 'email', dataType: tableau.dataTypeEnum.string },
			{ id: 'phone', dataType: tableau.dataTypeEnum.string },
			{ id: 'activated', dataType: tableau.dataTypeEnum.bool },
			{ id: 'last_login', dataType: tableau.dataTypeEnum.date },
			{ id: 'created', dataType: tableau.dataTypeEnum.date },
			{ id: 'modified', dataType: tableau.dataTypeEnum.date },
			{ id: 'signup_flow_variation', dataType: tableau.dataTypeEnum.string },
			{ id: 'has_created_company', dataType: tableau.dataTypeEnum.bool },
			{ id: 'is_admin', dataType: tableau.dataTypeEnum.int },
			{ id: 'timezone_name', dataType: tableau.dataTypeEnum.string },
			{ id: 'timezone_offset', dataType: tableau.dataTypeEnum.string },
			{ id: 'active_flag', dataType: tableau.dataTypeEnum.bool },
			{ id: 'role_id', dataType: tableau.dataTypeEnum.string },
			{ id: 'icon_url', dataType: tableau.dataTypeEnum.string },
			{ id: 'is_you', dataType: tableau.dataTypeEnum.bool },
		];

		var activitiesTable = { id: 'activities', alias: 'Activities', columns: activitiesCols };
		var dealsTable = { id: 'deals', alias: 'Deals', columns: dealsCols };
		var goalsTable = { id: 'goals', alias: 'Goals', columns: goalsCols };
		var notesTable = { id: 'notes', alias: 'Notes', columns: notesCols };
		var organizationsTable = { id: 'organizations', alias: 'Organizations', columns: organizationsCols };
		var personsTable = { id: 'persons', alias: 'Persons', columns: personsCols };
		var productsTable = { id: 'products', alias: 'Products', columns: productsCols };
		var stagesTable = { id: 'stages', alias: 'Stages', columns: stagesCols };
		var usersTable = { id: 'users', alias: 'Users', columns: usersCols };

		if (inputForm.tables.indexOf('activities') >= 0) callbackTables.push(activitiesTable);
		if (inputForm.tables.indexOf('deals') >= 0) callbackTables.push(dealsTable);
		if (inputForm.tables.indexOf('goals') >= 0) callbackTables.push(goalsTable);
		if (inputForm.tables.indexOf('notes') >= 0) callbackTables.push(notesTable);
		if (inputForm.tables.indexOf('organizations') >= 0) callbackTables.push(organizationsTable);
		if (inputForm.tables.indexOf('persons') >= 0) callbackTables.push(personsTable);
		if (inputForm.tables.indexOf('products') >= 0) callbackTables.push(productsTable);
		if (inputForm.tables.indexOf('stages') >= 0) callbackTables.push(stagesTable);
		if (inputForm.tables.indexOf('users') >= 0) callbackTables.push(usersTable);

		schemaCallback(callbackTables);
	};

	// Download the data
	myConnector.getData = function (table, doneCallback) {
		var inputForm = JSON.parse(tableau.connectionData);

		var tableData = [];
		var len;
		var lenj;
		var i = 0;
		var j = 0;
		var r = 0;
		var start = 0;
		var limit = 500;

		function formatURL(table, api_token, start, limit, start_date, end_date) {
			var url = 'https://api.pipedrive.com/v1/' + table + '?api_token=' + api_token + '&start=' + start + '&limit=' + limit;
			if(inputForm.startDate !== null) url = url + '&start_date=' + inputForm.startDate;
			if(inputForm.endDate !== null) url = url + '&end_date=' + inputForm.endDate;
			return url;
		}

		if (inputForm.tables.indexOf(table.tableInfo.id) >= 0) {
			var hasMore = true;

			while (hasMore) {
				$.ajax({
					url: formatURL(table.tableInfo.id, inputForm.apiKey, start, limit, inputForm.startDate, inputForm.endDate),
					dataType: 'json',
					async: false,
					success: function(resp) {
						r = r + 1;
						if (r > 100) return;

						hasMore = false;

						// This is the payload we care about
						var _data = resp.data;

						// This is pagination data (if available)
						var extra = resp.additional_data;

						if (extra && extra.pagination.more_items_in_collection) {
							hasMore = true;
							start = extra.pagination.next_start;
							//tableau.log('Has more rows, next start at ' + start);
						}

						for (i = 0, len = _data.length; i < len; i++) {
							var ndata = $.extend({}, _data[i]);

							if (table.tableInfo.id === 'activities') {
								var participants = _data[i].participants;
								if (participants) {
									ndata.participant_count = participants.length;
									ndata.participants = '';
									for (j = 0, lenj = _data[i].participants.length; j < lenj; j++) {
										ndata.participants = ndata.participants + _data[i].participants[j].person_id;
									}
									ndata.participants = ndata.participants.slice(0, -1);
								}
							}
							if (table.tableInfo.id === 'deals') {
								ndata.creator_user_id = _data[i].creator_user_id.value;
								ndata.user_id = _data[i].user_id.value;
								ndata.person_id = _data[i].person_id.value;
								ndata.creator_user_name = _data[i].creator_user_id.name;
								ndata.user_name = _data[i].user_id.name;
								ndata.person_name = _data[i].person_id.name;
							}
							if (table.tableInfo.id == 'persons') {
								if (_data[i].phone) {
									var home_phone = _data[i].phone.filter(function (phone) {
										return phone.label === 'home';
									});
									var work_phone = _data[i].phone.filter(function (phone) {
										return phone.label === 'work';
									});
									var mobile_phone = _data[i].phone.filter(function (phone) {
										return phone.label === 'mobile';
									});
									var other_phone = _data[i].phone.filter(function (phone) {
										return phone.label === 'other';
									});

									ndata.home_phone = home_phone.length > 0 ? home_phone[0].value : '';
									ndata.work_phone = work_phone.length > 0 ? work_phone[0].value : '';
									ndata.mobile_phone = mobile_phone.length > 0 ? mobile_phone[0].value : '';
									ndata.other_phone = other_phone.length > 0 ? other_phone[0].value : '';
								}
								if (_data[i].email) {
									var home_email = _data[i].email.filter(function (email) {
										return email.label === 'home';
									});
									var work_email = _data[i].email.filter(function (email) {
										return email.label === 'work';
									});
									var other_email = _data[i].email.filter(function (email) {
										return email.label === 'other';
									});

									ndata.home_email = home_email.length > 0 ? home_email[0].value : '';
									ndata.work_email = work_email.length > 0 ? work_email[0].value : '';
									ndata.other_email = other_email.length > 0 ? other_email[0].value : '';
								}

								ndata.has_im = (_data[i].im[0].value === '' ? false : true);

								// Need to add IM, but this is getting exhausting... homework for the first person that needs it.
							}
							if (table.tableInfo.id == 'products') {
								// At least one record, more if there are variations
								ndata.owner_id = _data[i].owner_id.id;
								ndata.price_amount = _data[i].prices[0].price;
								ndata.price_currency = _data[i].prices[0].currency;
								ndata.price_cost = _data[i].prices[0].cost;

								if (_data[i].product_variations) {
									for (j = 0, lenj = _data[i].product_variations.length; j < lenj; j++) {
										var mdata = $.extend({}, ndata);

										mdata.variation_id = _data[i].product_variations[j].id;
										mdata.variation_name = _data[i].product_variations[j].name;
										mdata.variation_price = _data[i].product_variations[j].prices[0].price;
										mdata.variation_commennt = _data[i].product_variations[j].prices[0].comment;

										tableData.push(mdata);
									}
								}
								else {
									tableData.push(ndata);
								}
							}
							else {
								tableData.push(ndata);
							}
						}

						table.appendRows(tableData);
					},
          error: function(resp) {
            hasMore = false;
          },
				});
			}
		}

		doneCallback();
	};

	tableau.registerConnector(myConnector);

	// Create event listeners for when the user submits the form
	$(document).ready(function () {
		$('#submitButton').click(function () {
			var inputForm = {
				apiKey: $('#api-key').val().trim(),
				startDate: $('#start-date').val().trim(),
				endDate: $('#end-date').val().trim(),
				tables: [],
			};

			if ($('#activities').is(':checked')) inputForm.tables.push('activities');
			if ($('#deals').is(':checked')) inputForm.tables.push('deals');
			if ($('#goals').is(':checked')) inputForm.tables.push('goals');
			if ($('#notes').is(':checked')) inputForm.tables.push('notes');
			if ($('#organizations').is(':checked')) inputForm.tables.push('organizations');
			if ($('#persons').is(':checked')) inputForm.tables.push('persons');
			if ($('#products').is(':checked')) inputForm.tables.push('products');
			if ($('#stages').is(':checked')) inputForm.tables.push('stages');
			if ($('#users').is(':checked')) inputForm.tables.push('users');

			if (inputForm.startDate === '') inputForm.startDate = null;
			if (inputForm.endDate === '') inputForm.endDate = null;
			if (inputForm.startDate !== null) inputForm.startDate = new Date(inputForm.startDate);
			if (inputForm.endDate !== null) inputForm.endDate = new Date(inputForm.endDate);

			function isValid(form) {
			  /*
			   * 0 = OK
         * 1 = invalid date format
         * 2 = start date is after end date
         * 3 = no API key
         */
				if ((form.startDate !== null && form.startDate.toString() === 'Invalid Date')
          || (form.endDate !== null && form.endDate.toString() === 'Invalid Date'))
					return 1;

				if (form.startDate !== null && form.endDate !== null) {
					// Check that start is earlier than finish
					if (form.startDate > form.endDate) return 2;
				}

				if(form.apiKey.length !== 40) return 3;

				return 0;
			}

			// Need validation?
      var valid = isValid(inputForm);
			if (valid === 0) {
				tableau.connectionData = JSON.stringify(inputForm); // Use this variable to pass data to your getSchema and getData functions
				tableau.connectionName = 'Pipedrive'; // This will be the data source name in Tableau
				tableau.submit(); // This sends the connector object to Tableau
			} else {
				if(valid === 1) $('#errorMsg').html('<strong>Invalid Date entered</strong>');
				else if(valid === 2) $('#errorMsg').html('<strong>End date must be after start date</strong>');
				else if(valid === 3) $('#errorMsg').html('<strong>Missing or incorrect API key</strong>');
			}
		});
	});
})();